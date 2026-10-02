"""Independently profile the baseline and penalized replay-order coefficient."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._provenance import build_script_provenance, file_sha256  # noqa: E402


def reference_weights(animal, session, pause):
    animals = set(animal)
    recordings = {a: set(session[animal == a]) for a in animals}
    pauses = {(a, s): set(pause[(animal == a) & (session == s)])
              for a in animals for s in recordings[a]}
    counts = {}
    for key in zip(animal, session, pause, strict=True):
        counts[key] = counts.get(key, 0) + 1
    return np.array([1 / (len(animals) * len(recordings[a]) * len(pauses[a, s]) * counts[a, s, p])
                     for a, s, p in zip(animal, session, pause, strict=True)])


def reference_predictions(bank, penalty=1.0):
    """Residualize order against unpenalized baseline, then solve one scalar ridge."""
    result = np.empty_like(bank["order"])
    baseline_result = np.empty_like(result)
    y = bank["post"] - bank["pre"]
    for animal in sorted(set(bank["animal"])):
        train = bank["animal"] != animal
        test = ~train
        w = reference_weights(bank["animal"][train], bank["session"][train], bank["pause"][train])
        b = bank["baseline"][train]
        scale = np.sqrt(np.einsum("i,ij,ij->j", w, b, b))
        scale[scale == 0] = 1
        design = b / scale
        root_w = np.sqrt(w)
        weighted = design * root_w[:, None]
        inverse = np.linalg.pinv(weighted)
        outcome = y[train] * root_w
        base_coefficient = inverse @ outcome
        base_residual = outcome - weighted @ base_coefficient
        test_baseline = bank["baseline"][test] / scale
        for condition in range(bank["order"].shape[1]):
            raw = bank["order"][train, condition]
            x_scale = np.sqrt(np.dot(w, raw * raw))
            if x_scale == 0:
                x_scale = 1
            x = raw / x_scale
            wx = x * root_w
            residual_x = wx - weighted @ (inverse @ wx)
            beta = np.dot(residual_x, base_residual) / (np.dot(residual_x, residual_x) + penalty)
            coefficients = inverse @ (outcome - beta * wx)
            result[test, condition] = test_baseline @ coefficients + beta * bank["order"][test, condition] / x_scale
            baseline_result[test, condition] = test_baseline @ base_coefficient
    return result, baseline_result


def close(a, b, label):
    if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Independent prediction differs: {label}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.results / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if any(manifest.get(x) is not False for x in
           ("real_recordings_used", "real_association_fit", "biological_calibration_complete",
            "biological_inference_authorized", "goal_complete")):
        raise ValueError("Development claim boundary changed")
    for name, digest in manifest["outputs_sha256"].items():
        if file_sha256(args.results / name) != digest:
            raise ValueError(f"Changed fixture output {name}")
    for name, digest in manifest["input_file_sha256"].items():
        if file_sha256(manifest["input_file_paths"][name]) != digest:
            raise ValueError(f"Changed frozen input {name}")
    with (args.results / "fixture_summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    with (args.results / "fixture_animals.csv").open(newline="") as stream:
        animals = list(csv.DictReader(stream))
    checked_rows = 0
    for row in summaries:
        name = row["fixture"]
        with np.load(args.results / f"{name}.npz", allow_pickle=False) as bank:
            pre, a, b = bank["pre"], bank["pre_rate_a"], bank["pre_rate_b"]
            features = np.stack([pre, np.log1p(a) - np.log1p(b), pre * np.log1p(a + b),
                                 np.log1p(bank["event_spikes_a"]) - np.log1p(bank["event_spikes_b"]),
                                 pre * bank["participation"]], axis=1)
            close(features, bank["baseline"], f"{name}/PRE baseline controls")
            informative = sum(bool(np.any(bank["order"][bank["animal"] == animal, 0] != 0))
                              for animal in set(bank["animal"]))
            if int(row["animals_with_nonzero_original_order"]) != informative or (
                    row["minimum_informative_animal_coverage_met"] != str(informative >= 3)):
                raise ValueError("Informative-animal denominator differs")
            prediction, baseline = reference_predictions(bank)
            close(prediction, bank["predictions"], f"{name}/all held-out predictions")
            close(baseline, bank["baseline_predictions"], f"{name}/all held-out baseline predictions")
            gain = (bank["post"][:, None] - pre[:, None] - baseline) ** 2 - (
                bank["post"][:, None] - pre[:, None] - prediction) ** 2
            values = []
            for animal in sorted(set(bank["animal"])):
                selected = bank["animal"] == animal
                weights = reference_weights(bank["animal"][selected], bank["session"][selected], bank["pause"][selected])
                actual = weights @ gain[selected]
                recorded = sorted([x for x in animals if x["fixture"] == name and x["animal"] == animal],
                                  key=lambda x: int(x["condition"]))
                if len(recorded) != 21:
                    raise ValueError("Animal/control denominator differs")
                close(actual, [float(x["heldout_mse_improvement"]) for x in recorded], f"{name}/{animal}/scores")
                values.append(actual)
            values = np.array(values)
            close(float(row["original_mean_animal_gain"]), values[:, 0].mean(), f"{name}/animal aggregation")
            close(float(row["original_minus_animal_median_shuffle_gain"]),
                  (values[:, 0] - np.median(values[:, 1:], axis=1)).mean(), f"{name}/shuffle contrast")
            checked_rows += prediction.size
        print(f"VERIFIED {name}: independently profiled every animal/control", flush=True)
    if len(summaries) != 4 or len(animals) != 4 * 4 * 21:
        raise ValueError("Incomplete fixture accounting")
    for row in summaries:
        gain = float(row["original_mean_animal_gain"])
        slope = float(row["mean_training_order_coefficient"])
        if row["fixture"] in {"existing_coordination_algebra", "nuisance_only_algebra"} and abs(gain) > 1e-12:
            raise ValueError("Baseline-only fixture acquired a spurious gain")
        if row["fixture"] == "planted_positive_order_algebra" and not (gain > 0 and slope > 0):
            raise ValueError("Positive fixture was not recovered")
        if row["fixture"] == "planted_negative_order_algebra" and not (gain > 0 and slope < 0):
            raise ValueError("Negative fixture direction was not recovered")
    verification = {**build_script_provenance(input_paths={"manifest": manifest_path}),
                    "verified": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "independent_prediction_entries": checked_rows, "fixtures": 4,
                    "scope": "Independent baseline projection and scalar ridge solution for every algebraic fixture prediction and animal/control score. Not replay validation or six-generator spike-level calibration.",
                    "real_association_verified": False, "biological_calibration_verified": False,
                    "goal_complete": False}
    args.output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
