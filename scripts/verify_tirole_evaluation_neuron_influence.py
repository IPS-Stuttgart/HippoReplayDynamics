"""Independent likelihood, frozen-label and aggregation checks for neuron influence."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def permutations_without(original, remove):
    keep = [x for x in range(original.shape[1]) if x != remove]
    lookup = {v: k for k, v in enumerate(keep)}
    output = []
    for row in original:
        values = []
        for cell in keep:
            target = row[cell]
            while target == remove:
                target = row[target]
            values.append(lookup[target])
        output.append(values)
    return np.array(output)


def odds(counts, maps, valid, conditional):
    obs = counts[counts.sum(axis=1) > 0]
    if not len(obs):
        return np.nan
    ll = []
    for track in range(2):
        lam = np.maximum(maps[track], 1e-4)
        if conditional:
            lam = lam / lam.sum(axis=0)
        value = obs @ np.log(lam)
        if not conditional:
            value -= 0.02 * lam.sum(axis=0)
        value -= np.log(valid[track].sum())
        value[:, ~valid[track]] = -np.inf
        ll.append(value)
    ll = np.stack(ll, axis=1)
    post = np.exp(ll - logsumexp(ll, axis=(1, 2), keepdims=True))
    mass = post.sum(axis=(0, 2))
    return float(np.log(max(mass[0], 1e-300)) - np.log(max(mass[1], 1e-300)))


def reference_interval(v, codes, weights):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    per_event = weights[:, codes]
    denominator = per_event[:, ok].sum(axis=1)
    draw = np.divide(per_event[:, ok] @ v[ok], denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
    finite = np.isfinite(draw)
    blocks = len(set(codes[ok]))
    ci = np.quantile(draw[finite], [0.025, 0.975]) if blocks >= 2 and finite.mean() >= 0.95 else [np.nan, np.nan]
    return np.array([v[ok].mean() if ok.any() else np.nan, *ci, ok.sum(), blocks, finite.mean()])


def run(source, audit, output):
    if output.exists():
        raise ValueError("new verification required")
    m = json.loads((audit / "manifest.json").read_text())
    if m["status"] != "complete" or m["git_dirty"] or m["selection_changed"] or m["inference_or_detector_cells_changed"] or m["rate_groups_refitted"]:
        raise ValueError("invalid completed audit")
    for path, digest in m["output_sha256"].items():
        if sha(audit / path) != digest:
            raise ValueError("changed audit output")
    if sha(source / "evaluation_identity_control_v1/manifest.json") != m["identity_manifest_sha256"]:
        raise ValueError("changed original identity control")
    frames = []
    for item in m["source_score_manifests"]:
        p = Path(item["manifest"])
        if sha(p) != item["sha256"]:
            raise ValueError("changed source manifest")
        sm = json.loads(p.read_text())
        if sm["session"] not in m["primary_sessions"] or sm["candidate_stratum"] != "ripple":
            continue
        f = p.parent / "event_content_coverage_scores.csv"
        if sha(f) != sm["output_sha256"]:
            raise ValueError("changed sequence scores")
        d = pd.read_csv(f)
        frames.append(d[(d.arm == "real") & (d.epoch == "POST")])
    scores = pd.concat(frames)
    keys = ["session", "event_id", "split", "repeat"]
    full = scores[scores.fraction == 1].set_index(keys)
    half = scores[scores.fraction == 0.5].set_index(keys).reindex(full.index)
    lost = full[full.sequence_accepted & ~half.sequence_accepted].reset_index()
    membership = pd.read_csv(audit / "frozen_lost_membership.csv")
    if set(map(tuple, lost[keys].to_numpy())) != set(map(tuple, membership[keys].to_numpy())):
        raise ValueError("lost membership changed")
    if not np.array_equal(lost.set_index(keys).sort_index().inferred_track, membership.set_index(keys).sort_index().full_track):
        raise ValueError("full inference labels changed")
    readout = pd.read_csv(audit / "evaluation_neuron_by_event_split.csv")
    summary = pd.read_csv(audit / "evaluation_neuron_influence_summary.csv")
    event_table = pd.read_csv(audit / "evaluation_neuron_event_summary.csv")
    if len(readout) != m["n_rows"] or readout.duplicated(["session", "event_id", "split", "omitted_index"]).any():
        raise ValueError("invalid readout row coverage")
    errors = []
    likelihood_rows = 0
    interval_rows = 0
    event_values = 0

    def close(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if not np.allclose(a, b, atol=1e-9, rtol=0, equal_nan=True):
            raise ValueError("independent numerical reconstruction mismatch")
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            errors.append(float(np.abs(a[finite] - b[finite]).max()))

    for path, digest in m["bank_manifest_sha256"].items():
        bank = Path(path).parent
        if sha(path) != digest:
            raise ValueError("changed bank manifest")
        bm = json.loads(Path(path).read_text())
        session = bm["session"]
        for name, value in bm["outputs_sha256"].items():
            if sha(bank / name) != value:
                raise ValueError("changed bank")
        maps = np.load(bank / "RUN_maps.npz")
        arrays = np.load(bank / "event_counts.npz")
        parts = json.loads((bank / "partitions.json").read_text())
        events = pd.read_csv(bank / "candidate_events.csv").set_index("event_id")
        d = readout[readout.session == session]
        labels = membership[membership.session == session]
        metrics = [prefix + name for prefix in ["", "conditional_"] for name in ["observed_log_odds", "identity_excess_log_odds", "identity_z_log_odds"]]
        omissions = {-1} | {u for part in parts["splits"] for u in part["evaluation"]}
        sm = summary[summary.session == session]
        em = event_table[event_table.session == session]
        if sm.duplicated(["omitted_index", "metric"]).any() or set(zip(sm.omitted_index, sm.metric, strict=True)) != {(u, metric) for u in omissions for metric in metrics}:
            raise ValueError("incomplete summary omissions or metrics")
        if em.duplicated(["omitted_index", "event_id"]).any() or set(zip(em.omitted_index, em.event_id, strict=True)) != {
            (u, eid) for u in omissions for eid in labels.event_id.unique()
        }:
            raise ValueError("incomplete event aggregation")
        run_rate = (maps["rates"] * (maps["occupancy_s"] / maps["occupancy_s"].sum(axis=1, keepdims=True))[:, None]).sum(axis=2).mean(axis=0)
        for split, part in enumerate(parts["splits"]):
            cells = np.array(part["evaluation"])
            selected = set(labels.loc[labels.split == split, "event_id"])
            expected = {(eid, omit) for eid in selected for omit in [-1, *cells]}
            observed = d[d.split == split]
            if set(zip(observed.event_id, observed.omitted_index, strict=True)) != expected:
                raise ValueError("missing or extra neuron omission")
            rng = np.random.default_rng(seed(20260918, session, split, "rate-matched-evaluation-identity"))
            groups = np.array_split(np.argsort(run_rate[cells], kind="stable"), max(1, len(cells) // 3))
            perm = np.tile(np.arange(len(cells)), (199, 1))
            for group in groups:
                for row in perm:
                    row[group] = rng.permutation(group)
            for row in observed.sort_values("event_id").groupby("omitted_index").head(1).to_dict("records"):
                omit = int(row["omitted_index"])
                keep = np.arange(len(cells)) if omit == -1 else np.flatnonzero(cells != omit)
                null = perm if omit == -1 else permutations_without(perm, int(np.flatnonzero(cells == omit)[0]))
                use = cells[keep]
                eid = int(row["event_id"])
                counts = arrays["counts"][arrays["offsets"][eid] : arrays["offsets"][eid + 1]][:, use]
                close([len(use), counts.sum(), (counts.sum(axis=0) > 0).sum()], [row["n_evaluation_cells"], row["n_evaluation_spikes"], row["n_evaluation_active_cells"]])
                for conditional, prefix in [(False, ""), (True, "conditional_")]:
                    values = np.array([odds(counts, maps["rates"][:, use][:, v], maps["valid_bins"], conditional) for v in np.vstack([np.arange(len(use)), null])])
                    avg, sd = values[1:].mean(), values[1:].std(ddof=1)
                    z = (values[0] - avg) / sd if np.isfinite(sd) and sd > 1e-12 else np.nan
                    close(
                        [values[0], avg, sd, values[0] - avg, z],
                        [row[prefix + k] for k in ["observed_log_odds", "null_mean_log_odds", "null_sd_log_odds", "identity_excess_log_odds", "identity_z_log_odds"]],
                    )
                likelihood_rows += 1
        ids = events.index[(events.epoch == "POST") & events.primary_ripple_candidate]
        _, codes = np.unique(np.floor(events.loc[ids, "start_s"].to_numpy() / 60).astype(int), return_inverse=True)
        k = codes.max() + 1
        weights = np.random.default_rng(seed(20260918, session, "POST", "ripple_z3", 60, "occupied-time-block-bootstrap")).multinomial(k, np.full(k, 1 / k), size=2000)
        base = d[d.omitted_index == -1].set_index(["event_id", "split"])
        base_vectors = {}
        for omitted in sorted(set(summary.loc[summary.session == session, "omitted_index"])):
            replacement = d[d.omitted_index == omitted].set_index(["event_id", "split"])
            for metric in summary.metric.unique():
                values = {}
                for eid, group in labels.groupby("event_id"):
                    terms = []
                    for label in group.to_dict("records"):
                        key = (eid, int(label["split"]))
                        value = (replacement if key in replacement.index else base).loc[key, metric]
                        terms.append(value * (1 if label["full_track"] == 1 else -1))
                    finite = np.array(terms)[np.isfinite(terms)]
                    values[eid] = float(np.median(finite)) if len(finite) else np.nan
                vector = pd.Series(values).reindex(ids).to_numpy()
                if omitted == -1:
                    base_vectors[metric] = vector
                observed_events = event_table[(event_table.session == session) & (event_table.omitted_index == omitted)].set_index("event_id")
                close([values[eid] for eid in observed_events.index], observed_events[metric].to_numpy())
                event_values += len(observed_events)
                row = summary[(summary.session == session) & (summary.omitted_index == omitted) & (summary.metric == metric)].iloc[0]
                for prefix, v in [("", vector), ("change_", vector - base_vectors[metric])]:
                    close(
                        reference_interval(v, codes, weights),
                        [row[prefix + col] for col in ["point", "ci025", "ci975", "n_informative_events", "n_informative_blocks", "finite_bootstrap_fraction"]],
                    )
                interval_rows += 1
    result = {
        "status": "pass",
        "manifest_sha256": sha(audit / "manifest.json"),
        "verifier_sha256": sha(__file__),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "all_omission_rows_checked": len(readout),
        "fixed_lost_membership_rows": len(membership),
        "independent_likelihood_rows": likelihood_rows,
        "event_values_reconstructed": event_values,
        "summary_interval_rows_reconstructed": interval_rows,
        "max_absolute_error": max(errors),
        "scope": "All frozen memberships, omission coverage, event medians and interval rows; one direct observed/199-null likelihood reconstruction per omission per split. Not biological validation.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.source_dir, a.audit_dir, a.output)
