"""Training-only calibration extension with the original synthetic test anchors fixed."""

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
from hipporeplayimm.two_track_content import content_readout, stable_seed
from scripts.calibrate_tirole_content_measurement import generate_track_counts, ordered_path
from scripts.crossfit_tirole_composition_mixture import arrays_from, fit_mixture, likelihoods, target_weights, templates
from scripts.report_tirole_composition_calibration import probabilities
from scripts.report_tirole_measurement_calibration import check_rows
from scripts.score_tirole_content_coverage import validate_bank

TRAIN_PER_EPOCH = 80
N_BOOT = 2000
PRIORS = (0.25, 0.4, 0.5, 0.6, 0.75)


def select_training(events, test, session):
    """Only detector metadata determine new anchors; exclude old windows plus one second."""
    eligible = events.ripple_supported & ~events.event_id.isin(test.event_id)
    for e in test.itertuples():
        eligible &= ~((events.start_s < e.end_s + 1.0) & (events.end_s > e.start_s - 1.0))
    selected = []
    for _, group in events[eligible].groupby("epoch"):
        ordered = sorted(group.event_id, key=lambda e: stable_seed(20260918, session, e, "training-only-composition-anchor"))
        selected.extend(ordered[:TRAIN_PER_EPOCH])
    return events[events.event_id.isin(selected)].sort_values("event_id")


def simulate_training(info, parts, selected, counts, offsets, maps):
    rows = []
    for e in selected.itertuples():
        original = counts[offsets[e.event_id] : offsets[e.event_id + 1]]
        for split, part in enumerate(parts["splits"]):
            train = np.asarray(part["inference"])
            held = np.asarray(part["evaluation"])
            path = ordered_path(maps["valid_bins"], len(original), np.random.default_rng(stable_seed(20260918, info["session"], e.event_id, split, "known-path")))
            for track in range(2):
                rng = np.random.default_rng(stable_seed(20260918, info["session"], e.event_id, split, track, "known-track-spikes"))
                c = generate_track_counts(original, maps["rates"], train, held, path, track, rng)
                swaps = rng.integers(0, 2, (199, len(held))).astype(bool)
                values = {}
                for readout, conditional in [("poisson", False), ("conditional_count", True)]:
                    result = content_readout(c[:, held], maps["rates"][:, held], maps["valid_bins"], swaps, conditional)
                    values[readout] = result["track2_probability"]
                rows.append(
                    {
                        "session": info["session"],
                        "anchor_event": e.event_id,
                        "epoch": e.epoch,
                        "split": split,
                        "truth_track": track + 1,
                        "n_evaluation_spikes": int(c[:, held].sum()),
                        "true_path_span_cm": float(abs(maps["bin_centers_cm"][path[-1]] - maps["bin_centers_cm"][path[0]])),
                        **values,
                    }
                )
    return pd.DataFrame(rows)


def fit_training(q, weights):
    # Fold 1 contains no training anchor: reuse the frozen histogram implementation.
    return templates(q, np.zeros(len(q), int), weights)[:, 1]


def test_likelihoods(q, components):
    return likelihoods(q, np.zeros(len(q), int), components[:, None])


def separated_bootstrap(train_q, test_q, target, session, readout):
    trng = np.random.default_rng(stable_seed(20260918, session, readout, "independent-training-bootstrap"))
    qrng = np.random.default_rng(stable_seed(20260918, session, readout, "frozen-test-bootstrap"))
    results = {pi: [] for pi in target}
    for _ in range(N_BOOT // 100):
        tr = trng.multinomial(len(train_q), np.full(len(train_q), 1 / len(train_q)), size=100)
        te = qrng.multinomial(len(test_q), np.full(len(test_q), 1 / len(test_q)), size=100)
        ll = test_likelihoods(test_q, fit_training(train_q, tr)).reshape(100, -1, 2)
        for pi, weights in target.items():
            estimate, _ = fit_mixture(ll, (te[:, :, None, None] * weights).reshape(100, -1))
            results[pi].extend(estimate)
    return {pi: np.asarray(v) for pi, v in results.items()}


def run(bank, source, output):
    if output.exists():
        raise ValueError("new immutable calibration directory required")
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze protocol before execution")
    info, parts, events, counts, offsets, maps = validate_bank(bank)
    if info["cohort_stratum"] != "strict_RUN_pass":
        raise ValueError("this bounded follow-up is primary sessions only")
    m = json.loads((source / "manifest.json").read_text())
    if m["status"] != "complete" or m["git_dirty"] or m["bank_manifest_sha256"] != file_sha256(bank / "manifest.json"):
        raise ValueError("unverified or mismatched frozen calibration")
    for name, digest in m["output_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("changed frozen test source")
    test = pd.read_csv(source / "count_anchors.csv").sort_values("event_id")
    data = pd.read_csv(source / "known_track_calibration.csv")
    check_rows(data, pd.read_csv(source / "known_temporal_calibration.csv"), test)
    data = probabilities(data)
    train = select_training(events, test, info["session"])
    if len(train) < 5 or set(train.event_id) & set(test.event_id):
        raise ValueError("insufficient distinct training-only anchors")
    output.mkdir(parents=True)
    train.to_csv(output / "training_anchors.csv", index=False)
    test.to_csv(output / "frozen_test_anchors.csv", index=False)
    generated = simulate_training(info, parts, train, counts, offsets, maps)
    if len(generated) != len(train) * 10 or generated.duplicated(["anchor_event", "split", "truth_track"]).any():
        raise ValueError("incomplete training generator")
    generated.to_csv(output / "training_readouts.csv", index=False)
    ix = pd.MultiIndex.from_product([train.event_id, range(5), [1, 2]], names=["anchor_event", "split", "truth_track"])
    ordered = generated.set_index(ix.names).reindex(ix)
    rows, selections, bootrows, component_rows, llrows = [], [], [], [], []
    for readout in ["poisson", "conditional_count"]:
        train_q = ordered[readout].to_numpy().reshape(len(train), 5, 2)
        q, nspikes, span, full, half = arrays_from(data, test, readout)
        component = fit_training(train_q, np.ones((1, len(train))))
        ll = test_likelihoods(q, component)
        for s in range(5):
            for t in range(2):
                component_rows.append(
                    {
                        "readout": readout,
                        "split": s,
                        "truth_track": t + 1,
                        "n_unique_training_anchors": int(np.isfinite(train_q[:, s]).all(axis=1).sum()),
                        **{f"bin{k}": component[0, s, t, k] for k in range(5)},
                    }
                )
        for a, eid in enumerate(test.event_id):
            for s in range(5):
                for t in range(2):
                    llrows.append(
                        {
                            "anchor_event": eid,
                            "readout": readout,
                            "split": s,
                            "truth_track": t + 1,
                            "posterior_mass": q[a, s, t],
                            "likelihood_track1": ll[0, a, s, t, 0],
                            "likelihood_track2": ll[0, a, s, t, 1],
                        }
                    )
        primary = (test.epoch.eq("POST") & test.primary_ripple_candidate).to_numpy()[:, None] & np.ones((1, 5), bool)
        bootstrap = separated_bootstrap(train_q, q, {pi: target_weights(q, primary, prior=pi) for pi in [0.25, 0.5, 0.75]}, info["session"], readout)
        for pi, draws in bootstrap.items():
            bootrows.extend({"readout": readout, "true_pi": pi, "draw": i, "estimated_pi": x} for i, x in enumerate(draws))
        for epoch in ["PRE", "POST"]:
            for ripple in [False, True]:
                subset = ((test.epoch == epoch) & (test.primary_ripple_candidate == ripple)).to_numpy()[:, None] & np.ones((1, 5), bool)
                conditions = {
                    "all": np.ones(nspikes.shape, bool),
                    "count_1_to_4": (nspikes >= 1) & (nspikes <= 4),
                    "count_at_least_5": nspikes >= 5,
                    "path_under_100cm": span < 100,
                    "path_at_least_100cm": span >= 100,
                }
                for label, mask in conditions.items():
                    for pi in PRIORS:
                        w = target_weights(q, subset & mask, prior=pi)
                        estimate, complete = fit_mixture(ll.reshape(1, -1, 2), w.reshape(1, -1))
                        row = {
                            "session": info["session"],
                            "readout": readout,
                            "epoch": epoch,
                            "ripple_positive": ripple,
                            "condition": label,
                            "true_pi": pi,
                            "estimated_pi": estimate[0],
                            "error": estimate[0] - pi,
                            "complete_point_likelihood": complete[0],
                            "n_informative_anchors": int((w.sum(axis=(1, 2)) > 0).sum()),
                            "ci025": np.nan,
                            "ci975": np.nan,
                            "finite_bootstrap_fraction": np.nan,
                            "recovery_gate": "not_primary_target",
                        }
                        if epoch == "POST" and ripple and label == "all" and pi in bootstrap:
                            good = bootstrap[pi][np.isfinite(bootstrap[pi])]
                            row["finite_bootstrap_fraction"] = len(good) / N_BOOT
                            if len(good) >= 0.95 * N_BOOT and row["n_informative_anchors"] >= 2:
                                lo, hi = np.quantile(good, [0.025, 0.975])
                                row.update(ci025=lo, ci975=hi)
                                direction = hi < 0.5 if pi < 0.5 else lo > 0.5 if pi > 0.5 else True
                                row["recovery_gate"] = "pass" if complete[0] and np.isfinite(estimate[0]) and lo <= pi <= hi and direction else "fail"
                            else:
                                row["recovery_gate"] = "fail_insufficient_bootstrap"
                        rows.append(row)
                for group, accepted in [("full", full), ("retained", full & half), ("half", half)]:
                    w = target_weights(q, subset, accepted=accepted)
                    estimate, complete = fit_mixture(ll.reshape(1, -1, 2), w.reshape(1, -1))
                    truth = w[:, :, 1].sum() / w.sum() if w.sum() else np.nan
                    selections.append(
                        {
                            "session": info["session"],
                            "readout": readout,
                            "epoch": epoch,
                            "ripple_positive": ripple,
                            "group": group,
                            "true_weighted_pi": truth,
                            "estimated_pi": estimate[0],
                            "error": estimate[0] - truth,
                            "complete_likelihood": complete[0],
                            "n_informative_anchors": int((w.sum(axis=(1, 2)) > 0).sum()),
                        }
                    )
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("calibration code changed during execution")
    for name, records in {"components": component_rows, "test_likelihoods": llrows, "recovery": rows, "selection_stress": selections, "bootstrap": bootrows}.items():
        pd.DataFrame(records).to_csv(output / (name + ".csv"), index=False)
    manifest = {
        "status": "complete",
        "session": info["session"],
        "code_commit": commit,
        "git_dirty": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "bank_dir": str(bank.resolve()),
        "bank_manifest_sha256": file_sha256(bank / "manifest.json"),
        "source_manifest": str((source / "manifest.json").resolve()),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "training_anchors": len(train),
        "test_anchors": len(test),
        "training_only_extension": True,
        "real_data_corrected": False,
        "test_spikes_regenerated": False,
        "test_trajectories_regenerated": False,
        "counts_conditioned_on_recorded_totals": True,
        "known_map_not_known_trajectory": True,
        "n_histogram_bins": 5,
        "pseudocount": 0.5,
        "minimum_unique_training_anchors": 5,
        "n_bootstrap": N_BOOT,
        "bootstrap_training_and_test_anchors_separately": True,
        "command_line": sys.argv,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "session": info["session"], "training_anchors": len(train)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.bank_dir, a.calibration_dir, a.output_dir)
