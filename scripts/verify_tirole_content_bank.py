"""Independent raw-spike histogram and direct-likelihood audit (no producer imports)."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import logsumexp


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def direct_posterior(counts, rates, valid, conditional=False):
    logweights = []
    for track in range(2):
        lam = np.maximum(rates[track], 1e-4)
        if conditional:
            lam = lam / lam.sum(axis=0, keepdims=True)
        ll = counts @ np.log(lam)
        if not conditional:
            ll -= 0.02 * lam.sum(axis=0)[None, :]
        ll -= np.log(np.count_nonzero(valid[track]))
        ll[:, ~valid[track]] = -np.inf
        logweights.append(ll)
    logweights = np.stack(logweights, axis=1)
    return np.exp(logweights - logsumexp(logweights, axis=(1, 2), keepdims=True))


def direct_correlations(p, centers):
    t, x = np.meshgrid(np.arange(p.shape[0]), centers, indexing="ij")
    result = []
    for k in range(2):
        weights = p[:, k]
        if weights.sum() == 0:
            result.append(0.0)
            continue
        tm = np.average(t, weights=weights)
        xm = np.average(x, weights=weights)
        cov = np.average((t - tm) * (x - xm), weights=weights)
        vt = np.average((t - tm) ** 2, weights=weights)
        vx = np.average((x - xm) ** 2, weights=weights)
        result.append(abs(cov) / np.sqrt(vt * vx) if vt * vx > 1e-24 else 0.0)
    return np.array(result)


def run(root, bank, output, scores=None):
    if output.exists():
        raise ValueError("new verifier output path required")
    info = json.loads((bank / "manifest.json").read_text())
    session = info["session"]
    c = loadmat(root / f"{session}_extracted_clusters.mat", simplify_cells=True)["clusters"]
    order = np.argsort(c["spike_times"])
    times = np.asarray(c["spike_times"])[order]
    ids = np.asarray(c["spike_id"])[order]
    events = pd.read_csv(bank / "candidate_events.csv", float_precision="round_trip")
    data = np.load(bank / "event_counts.npz")
    maps = np.load(bank / "RUN_maps.npz")
    offsets = data["offsets"]
    counts = data["counts"]
    units = data["unit_ids"]
    mismatch = 0
    max_error = 0
    verified = 0
    for e in events.to_dict("records"):
        eid = int(e["event_id"])
        n = int(e["n_time_bins"])
        edges = e["start_s"] + np.arange(n + 1) * 0.02
        a, b = np.searchsorted(times, [edges[0], edges[-1]], side="left")
        expect = np.column_stack([np.histogram(times[a:b][ids[a:b] == u], edges)[0] for u in units])
        actual = counts[offsets[eid] : offsets[eid + 1]]
        if expect.shape != actual.shape:
            raise ValueError("bank shape mismatch")
        err = np.abs(expect - actual)
        mismatch += int(np.count_nonzero(err))
        max_error = max(max_error, int(err.max(initial=0)))
        verified += expect.size
    statistics = []
    if scores is not None:
        table = pd.read_csv(scores / "event_content_coverage_scores.csv")
        meta = json.loads((scores / "manifest.json").read_text())
        if meta["bank_manifest_sha256"] != digest(bank / "manifest.json") or meta["output_sha256"] != digest(scores / "event_content_coverage_scores.csv"):
            raise ValueError("score input/output hash mismatch")
        parts = json.loads((bank / "partitions.json").read_text())
        split = meta["split"]
        repeat = meta["repeat"]
        train = np.array(parts["splits"][split]["inference"])
        np.array(parts["splits"][split]["evaluation"])
        perm = np.random.default_rng(seed(20260918, session, split, repeat, "coverage")).permutation(train)
        for row in table.to_dict("records"):
            eid = int(row["event_id"])
            obs = counts[offsets[eid] : offsets[eid + 1]]
            use = np.sort(perm[: max(1, int(np.ceil(row["fraction"] * len(perm))))])
            rng = np.random.default_rng(seed(20260918, session, eid, split, "sequence-nulls"))
            shifts = rng.integers(0, 20, (499, 2, counts.shape[1]))
            timeperm = np.argsort(rng.random((499, len(obs))), axis=1)
            identity = rng.permutation(train)
            cc = obs[:, use] if row["arm"] == "real" else obs[:, identity[np.searchsorted(train, use)]]
            active = int((cc.sum(axis=0) > 0).sum())
            nonempty = cc.sum(axis=1) > 0
            eligible = active >= 5 and nonempty.sum() >= 5
            if eligible != row["sequence_eligible"]:
                raise ValueError("eligibility mismatch")
            if not eligible:
                continue
            rates = maps["rates"][:, use]
            valid = maps["valid_bins"]
            p = direct_posterior(cc, rates, valid)
            p[~nonempty] = 0
            real = direct_correlations(p, maps["bin_centers_cm"])
            tnull = np.array([direct_correlations(p[v], maps["bin_centers_cm"]) for v in timeperm])
            # Audit null significance on eight opportunity-valid rows independently.
            if len(statistics) >= 8:
                continue
            fnull = []
            for j in range(499):
                rolled = np.stack([np.stack([np.roll(rates[k, u], shifts[j, k, uid]) for u, uid in enumerate(use)]) for k in range(2)])
                fp = direct_posterior(cc, rolled, valid)
                fp[~nonempty] = 0
                fnull.append(direct_correlations(fp, maps["bin_centers_cm"]))
            pt = (1 + (tnull >= real - 1e-12).sum(axis=0)) / 500
            pf = (1 + (np.array(fnull) >= real - 1e-12).sum(axis=0)) / 500
            accepted = (pt < 0.025) & (pf < 0.025)
            best = int(np.argmax(np.where(accepted, real, -np.inf))) if accepted.any() else int(np.argmax(real))
            if bool(accepted.any()) != row["sequence_accepted"] or best + 1 != row["inferred_track"]:
                raise ValueError("classification mismatch")
            observed = np.array([row["track1_p_time"], row["track2_p_time"], row["track1_p_field"], row["track2_p_field"]])
            if not np.allclose(observed, np.r_[pt, pf], atol=1e-12):
                raise ValueError("null p-value mismatch")
            statistics.append({"event_id": eid, "fraction": row["fraction"], "arm": row["arm"], "max_correlation_error": float(abs(real[best] - row["best_weighted_correlation"]))})
        # All evaluation content readouts are fixed across coverage/negative-control arms.
        eval_columns = [col for col in table if col.startswith(("evaluation_", "conditional_evaluation_"))]
        for _, group in table.groupby("event_id"):
            if any(group[col].nunique(dropna=False) != 1 for col in eval_columns):
                raise ValueError("evaluation readout changes across inference conditions")
    result = {
        "status": "pass" if mismatch == 0 and (scores is None or len(statistics) > 0) and all(r["max_correlation_error"] < 1e-10 for r in statistics) else "fail",
        "session": session,
        "raw_input_sha256": digest(root / f"{session}_extracted_clusters.mat"),
        "bank_manifest_sha256": digest(bank / "manifest.json"),
        "events_recounted": len(events),
        "count_entries_checked": verified,
        "count_mismatch_entries": mismatch,
        "max_count_error": max_error,
        "independent_sequence_null_checks": statistics,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "all raw spike histograms; all eligibility and fixed-evaluation invariants; first eight eligible score rows direct Poisson/correlation/null reconstruction; not a biological validation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--bank-dir", required=True, type=Path)
    p.add_argument("--scores-dir", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.dataset_root, a.bank_dir, a.output, a.scores_dir)
