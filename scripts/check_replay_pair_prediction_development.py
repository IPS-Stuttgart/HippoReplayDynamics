"""Run count-preserving synthetic checks of the downstream prediction core.

The responses are planted algebraic fixtures, not measured or simulated RUN
spikes. This cannot pass the six-generator biological calibration gate.
"""
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
from scripts._replay_pair_prediction import (  # noqa: E402
    PairData, aggregate_event_orders, nuisance_features, prediction_check, require,
)
from scripts.audit_replay_order_run_coordination import (  # noqa: E402
    order_asymmetry, stable_seed, whole_bin_shuffles,
)

CASES = ("existing_coordination_algebra", "nuisance_only_algebra",
         "planted_positive_order_algebra", "planted_negative_order_algebra")


def write_csv(path, rows):
    require(bool(rows), "Empty fixture output")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_predictors(p, parent):
    rng = np.random.default_rng(p["seed"])
    fields = {name: [] for name in ("animal", "session", "pause", "unit_a", "unit_b",
                                   "pre", "baseline", "order", "pre_rate_a", "pre_rate_b",
                                   "event_spikes_a", "event_spikes_b", "participation")}
    count_checks = 0
    for rat in range(4):
        for day in range(2):
            for pause in range(2):
                identity = f"fixture/rat{rat}/day{day}/pause{pause}"
                events = []
                if not (day == 0 and pause == 0):
                    for e in range(2):
                        counts = rng.poisson(.1, size=(24, 8))
                        permutation = rng.permutation(8)
                        for j, unit in enumerate(permutation):
                            counts[2 * j:2 * j + 2, unit] += rng.poisson(3, size=2)
                        active = counts.sum(axis=0) > 0
                        counts, ids = counts[:, active], np.arange(8)[active]
                        width = .005
                        lo, hi = parent["order_min_lag_s"], parent["order_max_lag_s"]
                        controls = whole_bin_shuffles(counts, p["n_shuffles"],
                                                      stable_seed(p["seed"], f"{identity}/event{e}"))
                        for control in controls:
                            require(np.array_equal(control.sum(axis=0), counts.sum(axis=0))
                                    and sorted(map(tuple, control)) == sorted(map(tuple, counts)),
                                    "Synthetic control changed population snapshots")
                            count_checks += 1
                        events.append({"event_id": f"{identity}/event{e}", "validated_replay": True,
                                       "unit_ids": ids, "spike_counts": counts.sum(axis=0),
                                       "order": order_asymmetry(counts, width, lo, hi),
                                       "shuffle_order": np.stack([order_asymmetry(x, width, lo, hi)
                                                                   for x in controls])})
                aggregated = aggregate_event_orders(np.arange(8), events, p["n_shuffles"])
                a, b = aggregated["unit_a"], aggregated["unit_b"]
                rates = rng.uniform(5, 40, size=8)
                pre = rng.normal(0, .1, len(a)) + .3 * aggregated["order"][:, 0]
                baseline = nuisance_features(pre, rates[a], rates[b], aggregated["event_spikes_a"],
                                             aggregated["event_spikes_b"], aggregated["participation"])
                values = {"animal": np.repeat(f"fixture_rat{rat}", len(a)),
                          "session": np.repeat(f"fixture_rat{rat}/day{day}", len(a)),
                          "pause": np.repeat(identity, len(a)), "unit_a": a, "unit_b": b,
                          "pre": pre, "baseline": baseline, "order": aggregated["order"],
                          "pre_rate_a": rates[a], "pre_rate_b": rates[b],
                          "event_spikes_a": aggregated["event_spikes_a"],
                          "event_spikes_b": aggregated["event_spikes_b"],
                          "participation": aggregated["participation"]}
                for name, value in values.items():
                    fields[name].append(value)
    return {name: np.concatenate(value) for name, value in fields.items()}, count_checks


def fixture_response(name, fields):
    baseline = fields["baseline"]
    if name == CASES[0]:
        change = -.4 * fields["pre"]
    else:
        change = baseline @ np.array([-.4, .02, .01, .01, .05])
        if name == CASES[2]:
            change += 2 * fields["order"][:, 0]
        if name == CASES[3]:
            change -= 2 * fields["order"][:, 0]
    return fields["pre"] + change


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    p = json.loads(args.protocol.read_text())
    parent_path = ROOT / p["parent_protocol"]
    parent = json.loads(parent_path.read_text())
    require(p["real_association_enabled"] is False and parent["primary_analysis_enabled"] is False,
            "This entry point does not authorize a real association")
    require(p["n_shuffles"] == parent["order_shuffles"] == 20
            and p["intercept"] is False, "Frozen model/shuffle configuration changed")
    provenance = build_script_provenance(input_paths={"protocol": args.protocol,
                                                     "parent_protocol": parent_path})
    require(provenance["git_dirty"] is False and provenance["code_commit"] != "unavailable",
            "A clean committed checkout is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictors, count_checks = fixture_predictors(p, parent)
    summaries, animals, folds = [], [], []
    for name in CASES:
        fields = {**predictors, "post": fixture_response(name, predictors)}
        data = PairData(**{key: fields[key] for key in PairData.__dataclass_fields__})
        result = prediction_check(data, order_penalty=1)
        originals = [row["order_coefficient_original_units"] for row in result["folds"] if row["condition"] == 0]
        summary = {**result["summary"], "fixture": name,
                   "mean_training_order_coefficient": float(np.mean(originals)),
                   "real_recordings_used": False, "spike_level_calibration": False}
        summary["shuffle_mean_animal_gains"] = json.dumps(summary["shuffle_mean_animal_gains"])
        summaries.append(summary)
        animals.extend([{"fixture": name, **row} for row in result["by_animal"]])
        folds.extend([{"fixture": name, **row, "training_feature_rms": json.dumps(row["training_feature_rms"])}
                      for row in result["folds"]])
        np.savez_compressed(args.output_dir / f"{name}.npz", **fields,
                            predictions=result["predictions"], baseline_predictions=result["baseline_predictions"])
        print(f"FIXTURE {name}: {summary['original_mean_animal_gain']:.9g}; biological_calibration=False", flush=True)
    write_csv(args.output_dir / "fixture_summary.csv", summaries)
    write_csv(args.output_dir / "fixture_animals.csv", animals)
    write_csv(args.output_dir / "fixture_folds.csv", folds)
    files = ["fixture_summary.csv", "fixture_animals.csv", "fixture_folds.csv"] + [f"{x}.npz" for x in CASES]
    manifest = {**provenance, "protocol_id": p["protocol_id"], "seed": p["seed"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version, "numpy_version": np.__version__,
                "outputs_sha256": {name: file_sha256(args.output_dir / name) for name in files},
                "whole_bin_count_preservation_checks": count_checks,
                "response_source": "planted pair-level algebra; not measured or simulated RUN spike coordination",
                "simulated_event_labels": "oracle fixtures only; not independently validated real replay",
                "real_recordings_used": False, "real_association_fit": False,
                "biological_calibration_complete": False, "biological_inference_authorized": False,
                "goal_complete": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
