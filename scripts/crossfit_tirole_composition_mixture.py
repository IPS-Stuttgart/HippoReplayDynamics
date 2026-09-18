"""Cross-fitted calibrated mixtures on existing synthetic anchors, never real events."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import stable_seed
from scripts.report_tirole_composition_calibration import probabilities
from scripts.report_tirole_measurement_calibration import check_rows

PRIORS = (0.25, 0.4, 0.5, 0.6, 0.75)
N_BINS = 5
N_BOOT = 2000


def folds_for(anchors, session):
    folds = {}
    for _, g in anchors.groupby(["epoch", "primary_ripple_candidate"]):
        ids = sorted(g.event_id, key=lambda e: stable_seed(20260918, session, e, "mixture-calibration-fold"))
        folds.update({int(e): i % 5 for i, e in enumerate(ids)})
    return np.array([folds[int(e)] for e in anchors.event_id])


def templates(q, folds, anchor_weights):
    """Returns draw x fold x split x truth x score-bin distributions."""
    q = np.asarray(q, float)
    folds = np.asarray(folds, int)
    w = np.atleast_2d(np.asarray(anchor_weights, float))
    if q.ndim != 3 or q.shape[2] != 2 or folds.shape != (len(q),) or w.shape[1] != len(q) or (w < 0).any():
        raise ValueError("paired anchor/split/track arrays and weights required")
    support = np.isfinite(q).all(axis=2)
    if ((q[np.isfinite(q)] < 0) | (q[np.isfinite(q)] > 1)).any() or not np.isfinite(w).all() or not np.isin(folds, range(5)).all():
        raise ValueError("invalid mixture training values")
    bins = np.minimum(np.floor(np.nan_to_num(q) * N_BINS).astype(int), N_BINS - 1)
    indicators = np.eye(N_BINS)[bins] * support[:, :, None, None]
    out = np.full((len(w), 5, q.shape[1], 2, N_BINS), np.nan)
    for f in range(5):
        train = folds != f
        counts = np.einsum("ra,astb->rstb", w * train, indicators) + 0.5
        prob = counts / counts.sum(axis=3, keepdims=True)
        n_unique = np.einsum("ra,as->rs", ((w > 0) & train).astype(int), support.astype(int))
        prob[n_unique < 5] = np.nan
        out[:, f] = prob
    return out


def likelihoods(q, folds, component):
    bins = np.minimum(np.floor(np.nan_to_num(q) * N_BINS).astype(int), N_BINS - 1)
    out = np.full((len(component), *q.shape, 2), np.nan)
    for a in range(len(q)):
        for s in range(q.shape[1]):
            if np.isfinite(q[a, s]).all():
                for truth in range(2):
                    out[:, a, s, truth] = component[:, folds[a], s, :, bins[a, s, truth]]
    return out


def fit_mixture(likelihood, weights):
    """Concave bounded mixture likelihood, vectorized over bootstrap draws."""
    x = np.asarray(likelihood, float)
    w = np.asarray(weights, float)
    if x.ndim != 3 or x.shape[2] != 2 or w.shape != x.shape[:2] or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("draw x observation x component likelihoods and weights required")
    wanted = w > 0
    finite = np.isfinite(x).all(axis=2) & (x > 0).all(axis=2)
    complete = (~wanted | finite).all(axis=1) & (w.sum(axis=1) > 0)
    x = np.where(finite[:, :, None], x, 1.0)
    total = w.sum(axis=1)
    difference = x[:, :, 1] - x[:, :, 0]
    identifiable = (np.where(wanted, np.abs(difference), 0).max(axis=1) > 1e-12) & (total > 0) & complete

    def gradient(pi):
        return (w * difference / (x[:, :, 0] + pi[:, None] * difference)).sum(axis=1)

    lo = np.zeros(len(w))
    hi = np.ones(len(w))
    low = gradient(lo) <= 0
    high = gradient(hi) >= 0
    for _ in range(48):
        mid = (lo + hi) / 2
        increasing = gradient(mid) > 0
        lo = np.where(increasing, mid, lo)
        hi = np.where(increasing, hi, mid)
    pi = np.where(low, 0.0, np.where(high, 1.0, (lo + hi) / 2))
    return np.where(identifiable, pi, np.nan), complete


def target_weights(q, subset, prior=None, accepted=None):
    support = np.isfinite(q).all(axis=2) & np.asarray(subset, bool)
    n = support.sum(axis=1)
    base = np.divide(support, n[:, None], out=np.zeros_like(support, float), where=n[:, None] > 0)
    if prior is not None:
        return base[:, :, None] * np.array([1 - prior, prior])[None, None]
    if accepted is None or accepted.shape != q.shape:
        raise ValueError("known prior or paired selection required")
    return base[:, :, None] * 0.5 * accepted


def bootstrap_estimates(q, folds, targets, seed):
    rng = np.random.default_rng(seed)
    results = {pi: [] for pi in targets}
    for _ in range(N_BOOT // 100):
        w = rng.multinomial(len(q), np.full(len(q), 1 / len(q)), size=100)
        ll = likelihoods(q, folds, templates(q, folds, w)).reshape(100, -1, 2)
        for pi, target in targets.items():
            tw = (w[:, :, None, None] * target[None]).reshape(100, -1)
            estimates, _ = fit_mixture(ll, tw)
            results[pi].extend(estimates)
    return {pi: np.array(v) for pi, v in results.items()}


def arrays_from(data, anchors, readout):
    ids = anchors.event_id.tolist()
    arrays = []
    g = data[(data.generator == "ordered") & (data.fraction == 1)]
    for truth in [1, 2]:
        a = g[g.truth_track == truth].pivot(index="anchor_event", columns="split", values=readout).reindex(index=ids, columns=range(5))
        arrays.append(a.to_numpy())
    q = np.stack(arrays, axis=2)

    def array(column, fraction=1.0):
        v = data[(data.generator == "ordered") & (data.fraction == fraction)]
        return np.stack([v[v.truth_track == t].pivot(index="anchor_event", columns="split", values=column).reindex(index=ids, columns=range(5)).to_numpy() for t in [1, 2]], axis=2)

    counts = array("n_evaluation_spikes")
    if not np.array_equal(counts[:, :, 0], counts[:, :, 1]):
        raise ValueError("track simulations have unmatched count anchors")
    return q, counts[:, :, 0], array("true_path_span_cm")[:, :, 0], array("sequence_accepted").astype(bool), array("sequence_accepted", 0.5).astype(bool)


def run(source, output):
    if output.exists():
        raise ValueError("new immutable experiment directory required")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze estimator before outcomes")
    m = json.loads((source / "manifest.json").read_text())
    if m["status"] != "complete" or m["git_dirty"]:
        raise ValueError("incomplete source calibration")
    for name, digest in m["output_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("changed source")
    data = pd.read_csv(source / "known_track_calibration.csv")
    anchors = pd.read_csv(source / "count_anchors.csv").sort_values("event_id")
    check_rows(data, pd.read_csv(source / "known_temporal_calibration.csv"), anchors)
    data = probabilities(data)
    session = m["session"]
    folds = folds_for(anchors, session)
    rows, selection, components, scores, bootrows = [], [], [], [], []
    for readout in ["poisson", "conditional_count"]:
        q, counts, span, full, half = arrays_from(data, anchors, readout)
        fitted = templates(q, folds, np.ones((1, len(q))))
        ll = likelihoods(q, folds, fitted)
        for f in range(5):
            for s in range(5):
                for t in range(2):
                    components.append(
                        {
                            "session": session,
                            "readout": readout,
                            "fold": f,
                            "split": s,
                            "truth_track": t + 1,
                            "training_anchor_ids": json.dumps(anchors.event_id[(folds != f) & np.isfinite(q[:, s]).all(axis=1)].tolist()),
                            **{f"bin{k}": fitted[0, f, s, t, k] for k in range(N_BINS)},
                        }
                    )
        for a, eid in enumerate(anchors.event_id):
            for s in range(5):
                for t in range(2):
                    scores.append(
                        {
                            "session": session,
                            "readout": readout,
                            "anchor_event": int(eid),
                            "split": s,
                            "truth_track": t + 1,
                            "fold": int(folds[a]),
                            "posterior_mass": q[a, s, t],
                            "likelihood_track1": ll[0, a, s, t, 0],
                            "likelihood_track2": ll[0, a, s, t, 1],
                        }
                    )
        primary = (anchors.epoch.eq("POST") & anchors.primary_ripple_candidate).to_numpy()[:, None] * np.ones((1, 5), bool)
        is_primary = data.cohort_stratum.iloc[0] == "strict_RUN_pass"
        bootstrap = (
            bootstrap_estimates(
                q, folds, {pi: target_weights(q, primary, prior=pi) for pi in (0.25, 0.5, 0.75)}, stable_seed(20260918, session, readout, "refitted-mixture-bootstrap")
            )
            if is_primary
            else {}
        )
        for pi, draws in bootstrap.items():
            bootrows.extend({"session": session, "readout": readout, "true_pi": pi, "draw": i, "estimated_pi": v} for i, v in enumerate(draws))
        for epoch in ["PRE", "POST"]:
            for ripple in [False, True]:
                stratum = ((anchors.epoch == epoch) & (anchors.primary_ripple_candidate == ripple)).to_numpy()[:, None] * np.ones((1, 5), bool)
                for condition, mask in {
                    "all": np.ones(counts.shape, bool),
                    "count_1_to_4": (counts >= 1) & (counts <= 4),
                    "count_at_least_5": counts >= 5,
                    "path_under_100cm": span < 100,
                    "path_at_least_100cm": span >= 100,
                }.items():
                    subset = stratum & mask
                    for pi in PRIORS:
                        w = target_weights(q, subset, prior=pi)
                        estimate, complete = fit_mixture(ll.reshape(1, -1, 2), w.reshape(1, -1))
                        row = {
                            "session": session,
                            "animal": data.animal.iloc[0],
                            "cohort_stratum": data.cohort_stratum.iloc[0],
                            "readout": readout,
                            "epoch": epoch,
                            "ripple_positive": ripple,
                            "condition": condition,
                            "true_pi": pi,
                            "estimated_pi": estimate[0],
                            "error": estimate[0] - pi,
                            "n_informative_anchors": int((w.sum(axis=(1, 2)) > 0).sum()),
                            "complete_point_likelihood": bool(complete[0]),
                            "boundary_fit": bool(np.isfinite(estimate[0]) and estimate[0] in (0.0, 1.0)),
                            "ci025": np.nan,
                            "ci975": np.nan,
                            "finite_bootstrap_fraction": np.nan,
                            "recovery_gate": "not_primary_target",
                        }
                        if is_primary and epoch == "POST" and ripple and condition == "all" and pi in bootstrap:
                            draws = bootstrap[pi]
                            good = draws[np.isfinite(draws)]
                            row["finite_bootstrap_fraction"] = len(good) / N_BOOT
                            if len(good) >= 0.95 * N_BOOT and row["n_informative_anchors"] >= 2:
                                lo, hi = np.quantile(good, [0.025, 0.975])
                                row.update(ci025=lo, ci975=hi)
                                direction = hi < 0.5 if pi < 0.5 else lo > 0.5 if pi > 0.5 else True
                                row["recovery_gate"] = "pass" if complete[0] and np.isfinite(estimate[0]) and lo <= pi <= hi and direction else "fail"
                            else:
                                row["recovery_gate"] = "fail_insufficient_bootstrap"
                        rows.append(row)
                for label, accepted in [("full", full), ("retained", full & half), ("half", half)]:
                    w = target_weights(q, stratum, accepted=accepted)
                    estimate, complete = fit_mixture(ll.reshape(1, -1, 2), w.reshape(1, -1))
                    truth = float(w[:, :, 1].sum() / w.sum()) if w.sum() else np.nan
                    selection.append(
                        {
                            "session": session,
                            "readout": readout,
                            "epoch": epoch,
                            "ripple_positive": ripple,
                            "group": label,
                            "true_weighted_pi": truth,
                            "estimated_pi": estimate[0],
                            "error": estimate[0] - truth,
                            "n_informative_anchors": int((w.sum(axis=(1, 2)) > 0).sum()),
                            "effective_counterfactual_weight": float(w.sum()),
                            "complete_likelihood": bool(complete[0]),
                        }
                    )
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("estimator changed during execution")
    output.mkdir(parents=True)
    for name, frame in {
        "recovery": pd.DataFrame(rows),
        "selection_stress": pd.DataFrame(selection),
        "components": pd.DataFrame(components),
        "heldout_likelihoods": pd.DataFrame(scores),
        "bootstrap": pd.DataFrame(bootrows, columns=["session", "readout", "true_pi", "draw", "estimated_pi"]),
        "folds": pd.DataFrame({"session": session, "anchor_event": anchors.event_id, "fold": folds}),
    }.items():
        frame.to_csv(output / (name + ".csv"), index=False)
    manifest = {
        "status": "complete",
        "code_commit": commit,
        "git_dirty": False,
        "session": session,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest": str((source / "manifest.json").resolve()),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "n_anchor_bootstraps_requested_for_primary": N_BOOT,
        "bootstrap_rows_written": len(bootrows),
        "n_histogram_bins": N_BINS,
        "pseudocount": 0.5,
        "real_data_corrected": False,
        "known_labels_used_for_test_weighting_not_optimizer": True,
        "biological_confirmation": False,
        "command_line": sys.argv,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"session": session, "status": "complete"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.calibration_dir, a.output_dir)
