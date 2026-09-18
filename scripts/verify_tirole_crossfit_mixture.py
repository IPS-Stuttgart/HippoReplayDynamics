"""Independent histogram, fold-exclusion, brent-root and refitted-bootstrap audit."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def close(a, b):
    if not np.allclose(a, b, atol=1e-9, rtol=0, equal_nan=True):
        raise ValueError("independent mixture audit mismatch")
    d = np.abs(np.asarray(a) - np.asarray(b))
    return float(d[np.isfinite(d)].max(initial=0))


def reference_templates(q, folds, weights):
    result = np.full((5, q.shape[1], 2, 5), np.nan)
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for f in range(5):
        for s in range(q.shape[1]):
            keep = (folds != f) & (weights > 0) & np.isfinite(q[:, s]).all(axis=1)
            if keep.sum() < 5:
                continue
            for track in range(2):
                hist = np.histogram(q[keep, s, track], bins=edges, weights=weights[keep])[0] + 0.5
                result[f, s, track] = hist / hist.sum()
    return result


def reference_likelihoods(q, folds, templates):
    ll = np.full((*q.shape, 2), np.nan)
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for a in range(len(q)):
        for s in range(q.shape[1]):
            if np.isfinite(q[a, s]).all():
                for t in range(2):
                    bin_id = min(np.searchsorted(edges, q[a, s, t], side="right") - 1, 4)
                    ll[a, s, t] = templates[folds[a], s, :, bin_id]
    return ll


def reference_fit(likelihood, weights):
    ll = likelihood.reshape(-1, 2)
    w = weights.reshape(-1)
    use = w > 0
    if not use.any() or not np.isfinite(ll[use]).all() or (ll[use] <= 0).any():
        return np.nan, False
    a, b = ll[use].T
    w = w[use]
    delta = b - a
    if np.abs(delta).max() <= 1e-12:
        return np.nan, True

    def derivative(p):
        return float(np.dot(w, delta / ((1 - p) * a + p * b)))

    if derivative(0) <= 0:
        return 0.0, True
    if derivative(1) >= 0:
        return 1.0, True
    return float(brentq(derivative, 0, 1, xtol=1e-13)), True


def reference_weights(q, mask, pi=None, accepted=None):
    out = np.zeros_like(q)
    for a in range(len(q)):
        splits = np.flatnonzero(mask[a] & np.isfinite(q[a]).all(axis=1))
        if len(splits):
            for s in splits:
                out[a, s] = np.array([1 - pi, pi]) / len(splits) if pi is not None else 0.5 * accepted[a, s] / len(splits)
    return out


def failure_reason(likelihood, weights):
    """Describe invalid fits without supplying a replacement estimate."""
    used = weights > 0
    if not used.any():
        return "no_target_observations"
    x = likelihood[used]
    if not np.isfinite(x).all() or (x <= 0).any():
        return "insufficient_training_anchors"
    if np.abs(x[:, 1] - x[:, 0]).max() <= 1e-12:
        return "identical_components"
    return "valid"


def run(folder, output):
    if output.exists():
        raise ValueError("new verification artifact required")
    m = json.loads((folder / "manifest.json").read_text())
    if m["status"] != "complete" or m["git_dirty"] or m["real_data_corrected"] or m["n_histogram_bins"] != 5:
        raise ValueError("invalid experiment manifest")
    source = Path(m["source_manifest"])
    if digest(source) != m["source_manifest_sha256"]:
        raise ValueError("changed source")
    sm = json.loads(source.read_text())
    for base, manifest in [(folder, m), (source.parent, sm)]:
        for name, h in manifest["output_sha256"].items():
            if digest(base / name) != h:
                raise ValueError("changed input/output")
    raw = pd.read_csv(source.parent / "known_track_calibration.csv")
    raw["poisson"] = raw.evaluation_track2_probability
    raw["conditional_count"] = expit(-np.where(raw.truth_track == 1, 1, -1) * raw.conditional_true_signed_log_odds)
    anchors = pd.read_csv(source.parent / "count_anchors.csv").sort_values("event_id")
    foldmap = {}
    for _, g in anchors.groupby(["epoch", "primary_ripple_candidate"]):
        ids = sorted(g.event_id, key=lambda e: seed(20260918, m["session"], e, "mixture-calibration-fold"))
        foldmap.update({int(e): i % 5 for i, e in enumerate(ids)})
    folds = np.array([foldmap[int(e)] for e in anchors.event_id])
    saved = pd.read_csv(folder / "folds.csv").set_index("anchor_event").loc[anchors.event_id]
    close(folds, saved.fold.to_numpy())
    recovery = pd.read_csv(folder / "recovery.csv")
    selection = pd.read_csv(folder / "selection_stress.csv")
    components = pd.read_csv(folder / "components.csv")
    predictions = pd.read_csv(folder / "heldout_likelihoods.csv")
    bootstrap = pd.read_csv(folder / "bootstrap.csv")
    checks = {"templates": 0, "heldout_likelihoods": 0, "recovery_points": 0, "selection_points": 0, "primary_intervals": 0, "refitted_bootstrap_points": 0}
    maxerr = 0.0
    bootstrap_failures = {}
    ix = pd.MultiIndex.from_product([anchors.event_id, range(5), [1, 2]], names=["anchor_event", "split", "truth_track"])
    ordered = raw[(raw.generator == "ordered") & (raw.fraction == 1)].set_index(ix.names).reindex(ix)
    if len(ordered) != len(ix) or ordered.session.isna().any():
        raise ValueError("incomplete source ordering")
    shape = (len(anchors), 5, 2)
    counts = ordered.n_evaluation_spikes.to_numpy().reshape(shape)[:, :, 0]
    span = ordered.true_path_span_cm.to_numpy().reshape(shape)[:, :, 0]
    full = ordered.sequence_accepted.to_numpy().reshape(shape)
    half = raw[(raw.generator == "ordered") & (raw.fraction == 0.5)].set_index(ix.names).reindex(ix).sequence_accepted.to_numpy().reshape(shape)
    for readout in ["poisson", "conditional_count"]:
        q = ordered[readout].to_numpy().reshape(shape)
        params = reference_templates(q, folds, np.ones(len(q)))
        ll = reference_likelihoods(q, folds, params)
        cs = components[components.readout == readout]
        if len(cs) != 50 or cs.duplicated(["fold", "split", "truth_track"]).any():
            raise ValueError("incomplete component table")
        for r in cs.to_dict("records"):
            mask = (folds != r["fold"]) & np.isfinite(q[:, r["split"]]).all(axis=1)
            train = anchors.event_id[mask].tolist()
            if json.loads(r["training_anchor_ids"]) != train:
                raise ValueError("anchor leakage or changed training membership")
            maxerr = max(maxerr, close(params[r["fold"], r["split"], r["truth_track"] - 1], [r[f"bin{k}"] for k in range(5)]))
            checks["templates"] += 1
        ps = predictions[predictions.readout == readout].set_index(ix.names).reindex(ix)
        maxerr = max(maxerr, close(q.reshape(-1), ps.posterior_mass.to_numpy()))
        maxerr = max(maxerr, close(ll.reshape(-1, 2), ps[["likelihood_track1", "likelihood_track2"]].to_numpy()))
        checks["heldout_likelihoods"] += len(ps)
        for r in recovery[recovery.readout == readout].to_dict("records"):
            mask = ((anchors.epoch == r["epoch"]) & (anchors.primary_ripple_candidate == r["ripple_positive"])).to_numpy()[:, None] * np.ones((1, 5), bool)
            conditions = {
                "all": np.ones(counts.shape, bool),
                "count_1_to_4": (counts >= 1) & (counts <= 4),
                "count_at_least_5": counts >= 5,
                "path_under_100cm": span < 100,
                "path_at_least_100cm": span >= 100,
            }
            weights = reference_weights(q, mask & conditions[r["condition"]], r["true_pi"])
            estimated, complete = reference_fit(ll, weights)
            maxerr = max(maxerr, close(estimated, r["estimated_pi"]))
            if complete != r["complete_point_likelihood"] or int((weights.sum(axis=(1, 2)) > 0).sum()) != r["n_informative_anchors"]:
                raise ValueError("point coverage mismatch")
            checks["recovery_points"] += 1
            if r["recovery_gate"] != "not_primary_target":
                draws = bootstrap[(bootstrap.readout == readout) & (bootstrap.true_pi == r["true_pi"])].set_index("draw")
                if len(draws) != 2000 or set(draws.index) != set(range(2000)):
                    raise ValueError("incomplete bootstrap draws")
                good = draws.estimated_pi.dropna().to_numpy()
                maxerr = max(maxerr, close(len(good) / 2000, r["finite_bootstrap_fraction"]))
                if len(good) >= 1900:
                    lo, hi = np.quantile(good, [0.025, 0.975])
                    maxerr = max(maxerr, close([lo, hi], [r["ci025"], r["ci975"]]))
                    direction = hi < 0.5 if r["true_pi"] < 0.5 else lo > 0.5 if r["true_pi"] > 0.5 else True
                    gate = "pass" if complete and np.isfinite(estimated) and lo <= r["true_pi"] <= hi and direction else "fail"
                else:
                    gate = "fail_insufficient_bootstrap"
                    if not np.isnan([r["ci025"], r["ci975"]]).all():
                        raise ValueError("invalid-draw intervals should be missing")
                if gate != r["recovery_gate"]:
                    raise ValueError("advancement gate mismatch")
                checks["primary_intervals"] += 1
        for r in selection[selection.readout == readout].to_dict("records"):
            mask = ((anchors.epoch == r["epoch"]) & (anchors.primary_ripple_candidate == r["ripple_positive"])).to_numpy()[:, None] * np.ones((1, 5), bool)
            accepted = {"full": full, "retained": full & half, "half": half}[r["group"]]
            weights = reference_weights(q, mask, accepted=accepted)
            estimated, complete = reference_fit(ll, weights)
            truth = weights[:, :, 1].sum() / weights.sum() if weights.sum() else np.nan
            maxerr = max(maxerr, close([estimated, truth], [r["estimated_pi"], r["true_weighted_pi"]]))
            if complete != r["complete_likelihood"]:
                raise ValueError("selection coverage mismatch")
            checks["selection_points"] += 1
        if not bootstrap.empty:
            draws = np.random.default_rng(seed(20260918, m["session"], readout, "refitted-mixture-bootstrap")).multinomial(len(q), np.full(len(q), 1 / len(q)), size=2000)
            mask = (anchors.epoch.eq("POST") & anchors.primary_ripple_candidate).to_numpy()[:, None] * np.ones((1, 5), bool)
            reasons = {}
            for i in range(2000):
                w = draws[i]
                ref = reference_likelihoods(q, folds, reference_templates(q, folds, w))
                for pi in [0.25, 0.5, 0.75]:
                    weights = reference_weights(q, mask, pi) * w[:, None, None]
                    estimated, _ = reference_fit(ref, weights)
                    if pi == 0.5:
                        reason = failure_reason(ref, weights)
                        reasons[reason] = reasons.get(reason, 0) + 1
                    row = bootstrap[(bootstrap.readout == readout) & (bootstrap.true_pi == pi) & (bootstrap.draw == i)]
                    if len(row) != 1:
                        raise ValueError("bootstrap point missing")
                    maxerr = max(maxerr, close(estimated, row.estimated_pi.iloc[0]))
                    checks["refitted_bootstrap_points"] += 1
            bootstrap_failures[readout] = reasons
    if checks["recovery_points"] != 200 or checks["heldout_likelihoods"] != len(anchors) * 20:
        raise ValueError("vacuous or incomplete validation")
    result = {
        "status": "pass",
        "session": m["session"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": digest(folder / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "checks": checks,
        "max_absolute_error": maxerr,
        "bootstrap_fit_status_counts": bootstrap_failures,
        "scope": "all source/output hashes, folds, training membership, histograms, heldout likelihoods, recovery and selection points, primary interval/gate arithmetic and every refitted bootstrap draw; not biological validation",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.experiment_dir, a.output)
