"""Independent dense-matrix reconstruction of held-out forward forecasts."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_null(transition):
    values, vectors = np.linalg.eig(transition.T)
    pi = np.real(vectors[:, np.argmin(np.abs(values - 1))])
    pi /= pi.sum()
    if (pi <= 0).any() or not np.allclose(pi @ transition, pi, atol=1e-11, rtol=0):
        raise ValueError("invalid stationary distribution")
    stay = np.diag(transition)
    target = pi * (1 - stay)
    flow = np.ones_like(transition)
    np.fill_diagonal(flow, 0)
    for _ in range(100000):
        flow *= (target / flow.sum(axis=1))[:, None]
        flow *= (target / flow.sum(axis=0))[None, :]
        if max(np.abs(flow.sum(axis=1) - target).max(), np.abs(flow.sum(axis=0) - target).max()) < 1e-14:
            break
    else:
        raise ValueError("independent null balancing failed")
    null = flow / pi[:, None]
    np.fill_diagonal(null, stay)
    if not np.allclose(null.sum(axis=1), 1, atol=1e-10, rtol=0) or not np.allclose(pi @ null, pi, atol=1e-11, rtol=0):
        raise ValueError("null constraints failed")
    return null


def reference_ll(counts, emissions):
    p = emissions / emissions.sum(axis=0, keepdims=True)
    coefficient = gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1)
    return counts @ np.log(p) + coefficient[:, None]


def reference_scores(counts, train, held, fit, null, h):
    ll = reference_ll(counts[:, train], fit["probabilities"][train])
    initial, transition = fit["initial"], fit["transition"]

    def filtering(matrix):
        q = initial.copy()
        rows = []
        for i, l in enumerate(ll):
            if i:
                q = q @ matrix
            l = l + np.log(q)
            q = np.exp(l - logsumexp(l))
            rows.append(q.copy())
        return np.array(rows)

    q = filtering(transition)
    nq = filtering(null)
    forecast = {
        "dynamic": q[:-h] @ np.linalg.matrix_power(transition, h),
        "matched_shared": q[:-h] @ np.linalg.matrix_power(null, h),
        "matched_own": nq[:-h] @ np.linalg.matrix_power(null, h),
        "frozen": q[:-h],
        "no_history": np.array([initial @ np.linalg.matrix_power(transition, t) for t in range(h, len(ll))]),
    }
    target = counts[h:][:, held]
    held_ll = reference_ll(target, fit["probabilities"][held])
    scores = {name: float(logsumexp(np.log(p) + held_ll, axis=1)[target.sum(axis=1) > 0].sum()) for name, p in forecast.items()}
    scores["global"] = float(reference_ll(target, fit["global_probability"][held, None]).sum())
    return scores


def run(bank, scored, output):
    if output.exists():
        raise ValueError("new independent audit required")
    manifest = json.loads((scored / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["git_dirty"] or manifest["expected_rows"] != manifest["actual_rows"]:
        raise ValueError("incomplete scientific forecasts")
    if sha(bank / "manifest.json") != manifest["bank_manifest_sha256"]:
        raise ValueError("bank mismatch")
    bm = json.loads((bank / "manifest.json").read_text())
    for name, digest in bm["outputs_sha256"].items():
        if sha(bank / name) != digest:
            raise ValueError("changed bank")
    for name, digest in manifest["output_sha256"].items():
        if sha(scored / name) != digest:
            raise ValueError("changed forecast output")
    x = pd.read_csv(scored / "forecast_by_event_split.csv")
    events = pd.read_csv(bank / "candidate_events.csv").set_index("event_id")
    if len(x) != manifest["actual_rows"] or x.duplicated(["event_id", "split", "repeat", "fraction", "horizon_ms"]).any():
        raise ValueError("incorrect forecast row coverage")
    folds = json.loads((scored / "folds.json").read_text())
    parts = json.loads((bank / "partitions.json").read_text())
    arrays = np.load(bank / "event_counts.npz")
    counts, offsets = arrays["counts"], arrays["offsets"]
    selected = set(x.event_id)
    tested = []
    maxerr = 0.0
    nchecked = 0
    chosen = []
    null_errors = []
    labels = []
    for source in manifest["source_score_manifests"]:
        path = Path(source["manifest"])
        if sha(path) != source["sha256"]:
            raise ValueError("changed label provenance")
        sm = json.loads(path.read_text())
        csv = path.parent / "event_content_coverage_scores.csv"
        if sha(csv) != sm["output_sha256"]:
            raise ValueError("changed sequence scores")
        labels.append(pd.read_csv(csv))
    labels = pd.concat(labels, ignore_index=True)
    labels = labels[labels.arm == "real"].set_index(["event_id", "split", "repeat", "fraction"])
    keys = list(zip(x.event_id, x.split, x.repeat, x.fraction, strict=True))
    expected = labels.loc[keys, "sequence_accepted"].to_numpy()
    if not np.array_equal(x.sequence_accepted, expected):
        raise ValueError("forecast selection differs from frozen sequence labels")
    for record in folds:
        fold = record["fold"]
        test = set(record["test_ids"])
        cal = set(record["calibration_ids"])
        if test & cal or not test or not cal or not test <= selected or not cal <= selected:
            raise ValueError("event-fold leakage")
        for eid in test:
            t = events.loc[eid]
            c = events.loc[sorted(cal)]
            if not ((c.end_s + 1 <= t.start_s) | (c.start_s >= t.end_s + 1)).all():
                raise ValueError("temporal guard violated")
        tested.extend(test)
        if set(x.loc[x.fold == fold, "event_id"]) != test:
            raise ValueError("scores use wrong fit fold")
        fit = np.load(scored / f"fit_{fold}.npz")
        pop = fit["population"]
        if set(pop) & set(parts["detector"]):
            raise ValueError("detector cells used in forecasting")
        null = reference_null(fit["transition"])
        null_errors.append(float(np.abs(null.sum(axis=1) - 1).max()))
        chosen.extend(x[x.fold == fold].groupby(["split", "fraction", "horizon_ms", "lost_with_thinning"], group_keys=False).head(1).to_dict("records"))
        for row in [v for v in chosen if v["fold"] == fold]:
            eid = int(row["event_id"])
            part = parts["splits"][int(row["split"])]
            original = np.array(part["inference"])
            held = np.searchsorted(pop, part["evaluation"])
            if row["fraction"] == 1:
                inference = original
            else:
                seed_parts = [20260918, bm["session"], int(row["split"]), int(row["repeat"]), "coverage"]
                seed = int.from_bytes(hashlib.sha256("|".join(map(str, seed_parts)).encode()).digest()[:8], "little")
                order = np.random.default_rng(seed).permutation(original)
                inference = np.sort(order[: int(np.ceil(len(order) * 0.5))])
            train = np.searchsorted(pop, inference)
            if set(train) & set(held):
                raise ValueError("cell-role leakage")
            c = counts[offsets[eid] : offsets[eid + 1]][:, pop]
            values = reference_scores(c, train, held, fit, null, int(row["horizon_ms"] // 20))
            for name, value in values.items():
                err = abs(value - row["score_" + name])
                maxerr = max(maxerr, err)
                if err > 1e-7 * (1 + abs(value)):
                    raise ValueError("independent predictive score mismatch")
                nchecked += 1
    if len(tested) != len(set(tested)) or set(tested) != selected or not nchecked:
        raise ValueError("vacuous or incomplete event-fold audit")
    result = {
        "status": "pass",
        "session": bm["session"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": sha(scored / "manifest.json"),
        "bank_manifest_sha256": sha(bank / "manifest.json"),
        "verifier_sha256": sha(Path(__file__)),
        "all_forecast_rows_checked": len(x),
        "events_with_disjoint_guarded_folds": len(selected),
        "independently_recomputed_rows": len(chosen),
        "independently_recomputed_model_scores": nchecked,
        "max_absolute_score_error": maxerr,
        "max_independent_null_row_error": max(null_errors),
        "scope": "all hashes, row identities, sequence labels and guarded folds; stratified score subset via independent dense filtering and IPFP null; not biological validation",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--scores-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.bank_dir, a.scores_dir, a.output)
