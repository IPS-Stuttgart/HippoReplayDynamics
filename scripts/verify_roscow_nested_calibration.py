#!/usr/bin/env python3
"""Independently verify nested task predictions, not a biological hypothesis."""

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
from calibrate_roscow_task_arm_readout import aggregate, evaluate

KINDS = ("poisson", "composition", "count_only")
GRID = np.array([1, 2, 4, 8, 16, 32, 64, 128, np.inf])


def independent_logits(x, y, training, target, exposure, prior_seconds, floor):
    pooled = x[training].sum(axis=0) / (len(training) * exposure)
    rates = []
    for arm in range(3):
        members = [j for j in training if y[j] == arm]
        rates.append(np.maximum((x[members].sum(axis=0) + prior_seconds * pooled)
                                / (len(members) * exposure + prior_seconds), floor))
    rates = np.asarray(rates)
    total = rates.sum(axis=1)
    return {
        "poisson": (x[target] * np.log(rates)).sum(axis=1) - exposure * total,
        "composition": (x[target] * np.log(rates / total[:, None])).sum(axis=1),
        "count_only": x[target].sum() * np.log(total) - exposure * total,
    }


def independent_nested(x, y, exposure, prior_seconds=1.0, floor=1e-10):
    prediction = {k: [] for k in KINDS}
    temperature = {k: [] for k in KINDS}
    for outer in range(len(y)):
        train = [j for j in range(len(y)) if j != outer]
        inner = {k: [] for k in KINDS}
        for heldout in train:
            fitting = [j for j in train if j != heldout]
            value = independent_logits(x, y, fitting, heldout, exposure, prior_seconds, floor)
            for kind in KINDS:
                inner[kind].append(value[kind])
        outer_logits = independent_logits(x, y, train, outer, exposure, prior_seconds, floor)
        for kind in KINDS:
            value = np.asarray(inner[kind])
            quality = []
            for t in GRID:
                scaled = value / t
                logp = scaled - logsumexp(scaled, axis=1, keepdims=True)
                quality.append(logp[np.arange(len(train)), y[train]].mean())
            t = GRID[np.flatnonzero(np.asarray(quality) >= max(quality) - 1e-12)[-1]]
            scaled = outer_logits[kind] / t
            prediction[kind].append(scaled - logsumexp(scaled))
            temperature[kind].append(t)
    return ({k: np.asarray(v) for k, v in prediction.items()},
            {k: np.asarray(v) for k, v in temperature.items()})


def checked_manifest(folder):
    m = json.loads((folder / "manifest.json").read_text())
    for name, digest in m["outputs"].items():
        if Path(name).name != name or file_sha256(folder / name) != digest:
            raise ValueError(f"Changed artifact: {name}")
    for name, path in m["input_file_paths"].items():
        if file_sha256(path) != m["input_file_sha256"][name]:
            raise ValueError(f"Changed input/code: {name}")
    return m


def verify(folder):
    m = checked_manifest(folder)
    source = Path(m["input_file_paths"]["source_manifest"]).parent
    source_manifest = checked_manifest(source)
    params = m["parameters"]
    exposure = params["window_stop_s"] - params["window_start_s"]
    arrays = np.load(m["input_file_paths"]["counts"], allow_pickle=False)
    sessions = pd.read_csv(m["input_file_paths"]["source_sessions"])
    sessions = sessions[sessions.status == "scored"]
    shifts = pd.read_csv(folder / "nested_calibration_shift_scores.csv")
    predictions = pd.read_csv(folder / "nested_calibration_trial_predictions.csv")
    key = ["animal", "session", "readout", "shift"]
    old = pd.read_csv(source / "roscow_task_arm_shift_controls.csv")
    cols = [*key, "balanced_accuracy"]
    pd.testing.assert_frame_equal(
        old[cols].sort_values(key).reset_index(drop=True),
        shifts[shifts.method == "uncalibrated"][cols].sort_values(key).reset_index(drop=True),
        check_dtype=False, atol=1e-9, rtol=1e-9)
    assert not shifts.duplicated([*key, "method"]).any()
    assert not predictions.duplicated(["animal", "session", "readout", "method", "trial_index_1based"]).any()
    assert len(shifts) == len(predictions) == int(sessions.n_trials.sum()) * 6
    clipping_differences = []
    for row in sessions.itertuples():
        prefix = f"{row.animal}_session{row.session}"
        x, y, trials = (arrays[prefix + suffix] for suffix in ("_counts", "_labels", "_trials"))
        prob, temps = independent_nested(x, y, exposure, params["prior_seconds"], params["rate_floor_hz"])
        # Reproduce old probability-floor scoring as well as exact log scoring.
        for shift in range(len(y)):
            shifted = np.roll(y, shift)
            logs = {k: [] for k in KINDS}
            for heldout in range(len(y)):
                training = [j for j in range(len(y)) if j != heldout]
                values = independent_logits(x, shifted, training, heldout, exposure,
                                            params["prior_seconds"], params["rate_floor_hz"])
                for kind in KINDS:
                    logs[kind].append(values[kind][shifted[heldout]] - logsumexp(values[kind]))
            for kind in KINDS:
                exact = float(np.mean(logs[kind]) + np.log(3))
                clipped = float(np.maximum(logs[kind], np.log(np.finfo(float).tiny)).mean() + np.log(3))
                previous = old[(old.animal == row.animal) & (old.session == row.session)
                               & (old.readout == kind) & (old["shift"] == shift)].iloc[0]
                new = shifts[(shifts.animal == row.animal) & (shifts.session == row.session)
                             & (shifts.readout == kind) & (shifts["shift"] == shift)
                             & (shifts.method == "uncalibrated")].iloc[0]
                np.testing.assert_allclose(new.mean_log_score_above_chance, exact, atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(previous.mean_log_score_above_chance, clipped, atol=1e-9, rtol=1e-9)
                if not np.isclose(exact, clipped, atol=1e-9, rtol=1e-9):
                    clipping_differences.append({"animal": row.animal, "session": int(row.session),
                                                "readout": kind, "shift": shift,
                                                "old_clipped_log_score": clipped, "exact_log_score": exact})
        for kind in KINDS:
            for method in ("uncalibrated", "nested_temperature"):
                sub = shifts[(shifts.animal == row.animal) & (shifts.session == row.session)
                             & (shifts.readout == kind) & (shifts.method == method)]
                assert sorted(sub["shift"]) == list(range(len(y)))
            pred = predictions[(predictions.animal == row.animal) & (predictions.session == row.session)
                               & (predictions.readout == kind) & (predictions.method == "nested_temperature")]
            assert pred.trial_index_1based.tolist() == trials.tolist()
            assert pred.chosen_arm_1based.tolist() == (y + 1).tolist()
            expected = ["uniform" if np.isinf(t) else str(int(t)) for t in temps[kind]]
            assert pred.temperature.astype(str).tolist() == expected
            np.testing.assert_allclose(pred[["p_arm1", "p_arm2", "p_arm3"]], np.exp(prob[kind]), atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(pred.true_arm_log_probability, prob[kind][np.arange(len(y)), y], atol=1e-10, rtol=1e-9)
            actual = sub[sub["shift"] == 0].iloc[0]
            for metric, value in evaluate(prob[kind], y).items():
                np.testing.assert_allclose(actual[metric], value, atol=1e-10, rtol=1e-9)
            np.testing.assert_allclose(actual.uniform_fraction, np.isinf(temps[kind]).mean())
    summary = aggregate(shifts)
    pd.testing.assert_frame_equal(summary, pd.read_csv(folder / "nested_calibration_by_animal.csv"), check_dtype=False)
    pd.testing.assert_frame_equal(shifts[shifts["shift"] == 0].reset_index(drop=True),
                                 pd.read_csv(folder / "nested_calibration_session_scores.csv"), check_dtype=False)
    primary = summary[(summary.method == "nested_temperature") & summary.readout.isin(["poisson", "composition"])]
    passed = bool(len(primary) == 6 and primary.useful_probability_screen.all())
    assert m["nested_probability_screen_passed"] == passed
    assert m["original_task_arm_screen_passed"] == source_manifest["task_arm_feasibility_passed"]
    assert not m["rest_analysis_authorized"] and not m["biological_hypothesis_scored"]
    result = {
        "status": "passed", "sessions_independently_recomputed": len(sessions),
        "original_trial_readout_predictions_independently_recomputed": int(sessions.n_trials.sum()) * 3,
        "uncalibrated_shift_readouts_independently_recomputed": len(old),
        "old_clipped_scores_reproduced": True, "log_probability_floor_differences": clipping_differences,
        "nested_shift_readouts_completeness_checked": int((shifts.method == "nested_temperature").sum()),
        "nested_shift_predictions_independently_recomputed": False,
        "animal_null_aggregation_recomputed": True,
        "nested_probability_screen_passed": passed, "new_biological_result": False,
        "verification_script_sha256": file_sha256(Path(__file__)),
        "result_manifest_sha256": file_sha256(folder / "manifest.json"),
    }
    (folder / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().result_dir), indent=2))

