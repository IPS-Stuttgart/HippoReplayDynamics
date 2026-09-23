#!/usr/bin/env python3
"""Verify frozen task readout from saved count matrices; audit probability quality."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import file_sha256
from validate_roscow_task_arm_readout import READOUTS, score_session, summarize_animals


def independent_probabilities(counts, labels, exposure, prior_seconds, floor):
    result = {kind: [] for kind in READOUTS}
    for heldout in range(len(labels)):
        training = [j for j in range(len(labels)) if j != heldout]
        pooled = np.sum(counts[training], axis=0) / (len(training) * exposure)
        rates = []
        for arm in range(3):
            members = [j for j in training if labels[j] == arm]
            rate = (np.sum(counts[members], axis=0) + prior_seconds * pooled) / (len(members) * exposure + prior_seconds)
            rates.append(np.maximum(rate, floor))
        rates = np.asarray(rates)
        x = counts[heldout]
        total = rates.sum(axis=1)
        for kind in READOUTS:
            if kind == "composition":
                value = np.sum(x * np.log(rates / total[:, None]), axis=1)
            elif kind == "count_only":
                value = x.sum() * np.log(total) - exposure * total
            else:
                value = np.sum(x * np.log(rates), axis=1) - exposure * total
            result[kind].append(np.exp(value - logsumexp(value)))
    return {kind: np.asarray(rows) for kind, rows in result.items()}


def verify(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        if Path(name).name != name or file_sha256(folder / name) != digest:
            raise ValueError(f"changed artifact: {name}")
    for name, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][name]:
            raise ValueError(f"changed input or code: {name}")
    counts = np.load(folder / "task_arm_count_matrices.npz", allow_pickle=False)
    preds = pd.read_csv(folder / "roscow_task_arm_trial_predictions.csv")
    stored_nulls = pd.read_csv(folder / "roscow_task_arm_shift_controls.csv")
    sessions = pd.read_csv(folder / "roscow_task_arm_sessions.csv")
    params = manifest["parameters"]
    exposure = params["window_stop_s"] - params["window_start_s"]
    frames, calibration = [], []
    for row in sessions[sessions.status == "scored"].itertuples():
        prefix = f"{row.animal}_session{row.session}"
        x, y = counts[prefix + "_counts"], counts[prefix + "_labels"]
        metric, _ = score_session(x, y, exposure)
        metric["animal"], metric["session"] = row.animal, row.session
        frames.append(metric)
        independent = independent_probabilities(x, y, exposure, params["prior_seconds"], params["rate_floor_hz"])
        for kind, prob in independent.items():
            subset = preds[(preds.animal == row.animal) & (preds.session == row.session) & (preds.readout == kind)]
            assert subset.trial_index_1based.tolist() == counts[prefix + "_trials"].tolist()
            np.testing.assert_allclose(subset[["p_arm1", "p_arm2", "p_arm3"]], prob, rtol=1e-8, atol=1e-12)
            truth = prob[np.arange(len(y)), y]
            calibration.append({"animal": row.animal, "session": row.session, "readout": kind,
                                "n_trials": len(y), "mean_max_probability": float(prob.max(axis=1).mean()),
                                "mean_log_score_above_chance": float(np.log(np.maximum(truth, np.finfo(float).tiny)).mean() + np.log(3)),
                                "fraction_true_arm_below_001": float((truth < 0.01).mean())})
    rerun = pd.concat(frames, ignore_index=True)
    keys = ["animal", "session", "readout", "shift"]
    pd.testing.assert_frame_equal(rerun.sort_values(keys).reset_index(drop=True),
                                  stored_nulls.sort_values(keys).reset_index(drop=True), check_dtype=False, rtol=1e-10, atol=1e-10)
    animals = summarize_animals(rerun)
    pd.testing.assert_frame_equal(animals, pd.read_csv(folder / "roscow_task_arm_by_animal.csv"), check_dtype=False)
    calibration = pd.DataFrame(calibration)
    calibration.to_csv(folder / "probability_quality_by_session.csv", index=False)
    report = {"status": "passed", "sessions_recomputed": len(frames), "all_shift_scores_recomputed": len(rerun),
              "independent_fold_formula_verified": True,
              "task_arm_feasibility_passed": manifest["task_arm_feasibility_passed"],
              "all_animal_readouts_below_uniform_log_score": bool((animals.mean_session_log_score_above_chance < 0).all()),
              "biological_hypothesis_scored": False,
              "verification_script_sha256": file_sha256(Path(__file__)),
              "calibration_csv_sha256": file_sha256(folder / "probability_quality_by_session.csv")}
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().result_dir), indent=2))


if __name__ == "__main__":
    main()
