"""Verify generated count endpoints and animal-balanced prediction development.

Refit the first fixed bank per generator independently; reconstruct all remaining
endpoints from saved means. This is not full biological calibration.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._provenance import build_script_provenance, file_sha256  # noqa: E402
from scripts.verify_replay_pair_prediction_development import close, reference_predictions, reference_weights  # noqa: E402
from scripts.verify_run_pair_coordination_endpoint import independent_coordination  # noqa: E402
from scripts.verify_run_pair_rate_glm_stress import independent_residuals  # noqa: E402


def reference_order(counts, width, parent):
    matrix = np.zeros((counts.shape[1], counts.shape[1]))
    for lag in range(1, len(counts)):
        if parent["order_min_lag_s"] - 1e-12 <= lag * width <= parent["order_max_lag_s"] + 1e-12:
            product = counts[:-lag].T @ counts[lag:]
            matrix += product - product.T
    totals = counts.sum(axis=0)
    denominator = np.outer(totals, totals)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)


def reference_marginal_mean(bank, period, generator, p, estimator):
    n = len(bank["unit_ids"])
    preference = np.arange(n) * 2 * np.pi / n
    theta_preference = preference.copy()
    if generator == "shared_theta_input" and period == "post":
        theta_preference[::2] += np.pi / 2
    position = bank[f"{period}_position"] - 24
    spatial = .3 / 24 * (position[:, 0, None] * np.cos(preference)
                         + position[:, 1, None] * np.sin(preference))
    log_rate = spatial + .2 * np.cos(bank[f"{period}_direction"][:, None] - preference)
    log_rate += .2 * np.log(bank[f"{period}_speed"][:, None] / 14.4)
    log_rate += .7 * np.cos(bank[f"{period}_theta"] - theta_preference)
    immigrants = estimator["run_bin_s"] * p["run_base_rate_hz"] * np.exp(log_rate)
    if generator == "rate_only_drift" and period == "post":
        immigrants[:, ::2] *= 2
        immigrants[:, 1::2] *= .5
    marginal = immigrants.copy()
    if generator == "existing_coordination_only" or (
            generator == "order_specific_update" and period == "post"):
        lag = round(p["coupling_lag_s"] / estimator["run_bin_s"])
        order = bank["generating_event_order"]
        for source, target in zip(order[:-1], order[1:], strict=True):
            marginal[lag:, target] += p["coupling_probability"] * immigrants[:-lag, source]
    return marginal


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify(args):
    manifest_path = args.results / "manifest.json"
    m = json.loads(manifest_path.read_text())
    for k in ("real_recordings_used", "real_association_fit", "replay_validation_complete",
              "biological_calibration_complete", "biological_inference_authorized", "goal_complete"):
        if m.get(k) is not False:
            raise ValueError(f"Development claim boundary changed: {k}")
    for name, digest in m["outputs_sha256"].items():
        if file_sha256(args.results / name) != digest:
            raise ValueError(f"Changed generated output {name}")
    for name, digest in m["input_file_sha256"].items():
        if file_sha256(m["input_file_paths"][name]) != digest:
            raise ValueError(f"Changed protocol {name}")
    p = json.loads(Path(m["input_file_paths"]["spike_protocol"]).read_text())
    parent = json.loads(Path(m["input_file_paths"]["parent_protocol"]).read_text())
    estimator = json.loads(Path(m["input_file_paths"]["endpoint_protocol"]).read_text())
    inventory, summaries = read_rows(args.results / "inventory.csv"), read_rows(args.results / "development_summary.csv")
    animals = read_rows(args.results / "animals.csv")
    expected_banks = p["animals"] * p["pauses_per_animal"] * len(p["generators"])
    if len(inventory) != expected_banks or len(summaries) != len(p["generators"]):
        raise ValueError("Incomplete generator/bank denominators")
    checked_periods, refitted_periods, checked_shuffles, checked_predictions = 0, 0, 0, 0
    for summary in summaries:
        generator = summary["generator"]
        with np.load(args.results / f"{generator}_predictions.npz", allow_pickle=False) as predictions:
            for row in [r for r in inventory if r["generator"] == generator]:
                with np.load(args.results / row["bank"], allow_pickle=False) as bank:
                    selected = predictions["pause"] == row["pause"]
                    a, b = predictions["unit_a"][selected], predictions["unit_b"][selected]
                    if not selected.any() or bank["unit_ids"].shape != (p["units"],):
                        raise ValueError("Changed population or omitted pause")
                    for period in ("pre", "post"):
                        counts = bank[f"{period}_counts"]
                        mean = bank[f"{period}_estimated_mean"]
                        oracle_mean = reference_marginal_mean(bank, period, generator, p, estimator)
                        close(oracle_mean, bank[f"{period}_oracle_mean"], "known marginal spike means")
                        times = bank[f"{period}_time_s"]
                        residual = (counts - mean) / np.sqrt(mean)
                        matrix, _ = independent_coordination(residual, times, estimator)
                        close(matrix, bank[f"{period}_coordination"], "physical-lag fitted matrix")
                        close(matrix[a, b], predictions[period][selected], "pair endpoint export")
                        oracle, _ = independent_coordination((counts - oracle_mean) / np.sqrt(oracle_mean), times, estimator)
                        close(oracle, bank[f"{period}_oracle_coordination"], "physical-lag oracle matrix")
                        if row["animal"] == "simulated_rat0" and row["pause"].endswith("pause0"):
                            covariates = {k: bank[f"{period}_{k}"] for k in ("position", "direction", "speed", "theta")}
                            _, independently_fitted = independent_residuals(counts, times, covariates, estimator,
                                                                           return_prediction=True, newton_refit=True)
                            np.testing.assert_allclose(independently_fitted, mean, rtol=1e-8, atol=1e-10)
                            refitted_periods += 1
                        checked_periods += 1
                    expected_order = np.zeros((p["n_shuffles"] + 1, p["units"], p["units"]))
                    if int(row["synthetic_events"]):
                        counts = bank["event_counts"]
                        expected_order[0] = reference_order(counts, p["event_bin_s"], parent)
                        digest = hashlib.sha256(f"{p['seed']}:{generator}/{row['pause']}".encode()).digest()
                        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                        for j in range(p["n_shuffles"]):
                            shuffled = counts[rng.permutation(len(counts))]
                            if sorted(map(tuple, shuffled)) != sorted(map(tuple, counts)):
                                raise ValueError("Shuffle changed population vectors")
                            expected_order[j + 1] = reference_order(shuffled, p["event_bin_s"], parent)
                            checked_shuffles += 1
                        close(expected_order[1:], bank["event_shuffle_order"], "event shuffle matrices")
                        close(counts.sum(axis=0)[a], predictions["event_spikes_a"][selected], "active-cell counts")
                    close(expected_order[:, a, b].T, predictions["order"][selected], "pause/control order export")
            pre = predictions["pre"]
            features = np.column_stack((pre, np.log1p(predictions["pre_rate_a"]) - np.log1p(predictions["pre_rate_b"]),
                pre * np.log1p(predictions["pre_rate_a"] + predictions["pre_rate_b"]),
                np.log1p(predictions["event_spikes_a"]) - np.log1p(predictions["event_spikes_b"]),
                pre * predictions["participation"]))
            close(features, predictions["baseline"], "mandatory PRE baseline features")
            fitted, baseline = reference_predictions(predictions, penalty=p["order_penalty"])
            close(fitted, predictions["predictions"], "every held-out prediction")
            close(baseline, predictions["baseline_predictions"], "every baseline prediction")
            gains = (predictions["post"][:, None] - pre[:, None] - baseline) ** 2 - (
                predictions["post"][:, None] - pre[:, None] - fitted) ** 2
            values = []
            for animal in sorted(set(predictions["animal"])):
                selected = predictions["animal"] == animal
                w = reference_weights(predictions["animal"][selected], predictions["session"][selected], predictions["pause"][selected])
                gain = w @ gains[selected]
                recorded = sorted([r for r in animals if r["generator"] == generator and r["animal"] == animal],
                                  key=lambda r: int(r["condition"]))
                if len(recorded) != p["n_shuffles"] + 1:
                    raise ValueError("Incomplete animal/control denominator")
                close(gain, [float(r["heldout_mse_improvement"]) for r in recorded], "animal/control gains")
                values.append(gain)
            values = np.array(values)
            close(float(summary["original_mean_animal_gain"]), values[:, 0].mean(), "animal-balanced gain")
            close(float(summary["original_minus_animal_median_shuffle_gain"]),
                  (values[:, 0] - np.median(values[:, 1:], axis=1)).mean(), "animal-balanced shuffle excess")
            checked_predictions += fitted.size
        print(f"VERIFIED {generator}: raw generated counts and all held-out predictions", flush=True)
    return {**build_script_provenance(input_paths={"manifest": manifest_path}), "verified": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "checked_periods": checked_periods,
        "independently_refitted_nuisance_periods": refitted_periods, "whole_bin_shuffles": checked_shuffles,
        "independent_prediction_entries": checked_predictions,
        "scope": "All generated count endpoints/order controls/predictions; nuisance GLMs independently refitted on the first fixed bank per generator only.",
        "full_biological_calibration_verified": False, "real_association_verified": False, "goal_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Do not overwrite a verification artifact")
    args.output.write_text(json.dumps(verify(args), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
