"""Compare verified RUN training-support sensitivities without fitting an association."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts._provenance import build_script_provenance, file_sha256
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256


KEY = ["animal", "session", "pause_id"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_endpoint(root, verification_path):
    manifest = json.loads((root / "manifest.json").read_text())
    verification = json.loads(verification_path.read_text())
    require(verification.get("verified") is True and
            verification["input_file_sha256"]["endpoint_manifest"] == file_sha256(root / "manifest.json"),
            "Identity-matched endpoint verification required")
    require(not manifest["association_fit"] and not verification["association_verified"] and
            not manifest["biological_calibration_complete"] and not verification["biological_calibration_verified"],
            "This reporter only compares uncalibrated nuisance endpoints")
    for name, digest in manifest["outputs_sha256"].items():
        require(file_sha256(root / name) == digest, f"Changed endpoint output {name}")
    for name, path in manifest["input_file_paths"].items():
        require(file_sha256(path) == manifest["input_file_sha256"][name], f"Changed endpoint input {name}")
    protocol = json.loads(Path(manifest["input_file_paths"]["protocol"]).read_text())
    source = json.loads(Path(manifest["input_file_paths"]["measurement_manifest"]).read_text())
    frames = {name: pd.read_csv(root / name, float_precision="round_trip") for name in
              ("period_quality.csv", "pauses.csv", "pairs.csv")}
    require(len(frames["pairs.csv"]) == manifest["dependent_pair_endpoints"] == verification["checked_pair_endpoints"],
            "Pair denominator differs from verification")
    require(len(frames["pauses.csv"]) == manifest["frozen_pauses"] == verification["checked_frozen_pauses"],
            "Pause denominator differs from verification")
    return manifest, protocol, source, frames


def compare_frames(old, new):
    pause_columns = KEY + ["status", "pair_endpoints"]
    pd.testing.assert_frame_equal(old["pauses.csv"][pause_columns].sort_values(KEY).reset_index(drop=True),
                                  new["pauses.csv"][pause_columns].sort_values(KEY).reset_index(drop=True))
    pair_key = KEY + ["unit_a", "unit_b"]
    for frames in (old, new):
        require(not frames["pairs.csv"].duplicated(pair_key).any(), "Duplicate pair identity")
        require(not frames["pairs.csv"].independent_biological_subject.any() and
                not frames["pairs.csv"].association_fit.any(), "Pair rows cannot be independent subjects")
    pd.testing.assert_frame_equal(old["pairs.csv"][pair_key].sort_values(pair_key).reset_index(drop=True),
                                  new["pairs.csv"][pair_key].sort_values(pair_key).reset_index(drop=True))
    period_key = KEY + ["period"]
    for frames in (old, new):
        require(not frames["period_quality.csv"].duplicated(period_key).any(), "Duplicate period identity")
        require(frames["period_quality.csv"].status.eq("measured").all(), "Unavailable periods cannot enter gain comparison")
    periods = old["period_quality.csv"].merge(new["period_quality.csv"], on=period_key,
                                            suffixes=("_matched", "_expanded"), validate="one_to_one", how="outer", indicator=True)
    require(periods._merge.eq("both").all() and len(periods) > 0, "Changed or empty period denominator")
    for name in ("source_bins", "usable_bins", "predicted_bins", "total_bins", "physical_lag_opportunities"):
        require(np.array_equal(periods[f"{name}_matched"], periods[f"{name}_expanded"]), f"Changed endpoint support: {name}")
    score = "heldout_poisson_improvement_over_global"
    for name in (f"{score}_matched", f"{score}_expanded"):
        require(np.isfinite(periods[name]).all(), "Nonfinite predictive score")
    periods["paired_predictive_gain_change"] = periods[f"{score}_expanded"] - periods[f"{score}_matched"]
    periods["matched_better_than_global"] = periods[f"{score}_matched"] > 0
    periods["expanded_better_than_global"] = periods[f"{score}_expanded"] > 0
    summaries = []
    for animal, rows in periods.groupby("animal", sort=True):
        both = rows.groupby(["session", "pause_id"]).expanded_better_than_global.all()
        summaries.append({"animal": animal, "periods": len(rows), "pauses": len(both),
                          "positive_periods_matched": int(rows.matched_better_than_global.sum()),
                          "positive_periods_expanded": int(rows.expanded_better_than_global.sum()),
                          "pauses_positive_in_both_periods": int(both.sum()),
                          "median_matched_predictive_gain": float(rows[f"{score}_matched"].median()),
                          "median_expanded_predictive_gain": float(rows[f"{score}_expanded"].median()),
                          "median_paired_predictive_gain_change": float(rows.paired_predictive_gain_change.median()),
                          "association_tested": False, "biological_calibration_complete": False})
    return periods.drop(columns="_merge"), pd.DataFrame(summaries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-endpoint", required=True, type=Path)
    parser.add_argument("--matched-verification", required=True, type=Path)
    parser.add_argument("--expanded-endpoint", required=True, type=Path)
    parser.add_argument("--expanded-verification", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    old_manifest, old_protocol, _, old = verified_endpoint(args.matched_endpoint, args.matched_verification)
    new_manifest, new_protocol, new_source, new = verified_endpoint(args.expanded_endpoint, args.expanded_verification)
    require(new_protocol.get("rate_training_support") == "same_period_full_run" and
            new_protocol.get("rate_score_comparator") == "matched_target_training_global", "Fixed matched comparator required")
    text = {"protocol_id", "frozen_before", "scope", "rate_model", "source_and_support", "claim_boundary"}
    require(set(new_protocol) - set(old_protocol) == {"rate_training_support", "rate_score_comparator"} and
            set(old_protocol) <= set(new_protocol) and
            all(new_protocol[k] == old_protocol[k] for k in set(old_protocol) - text), "Model or numeric settings changed")
    require(new_source["input_file_sha256"]["previous_measurement_manifest"] ==
            old_manifest["input_file_sha256"]["measurement_manifest"], "Different original source cohort")
    periods, animals = compare_frames(old, new)
    inputs = {"matched_manifest": args.matched_endpoint / "manifest.json", "matched_verification": args.matched_verification,
              "expanded_manifest": args.expanded_endpoint / "manifest.json", "expanded_verification": args.expanded_verification}
    provenance = build_script_provenance(input_paths=inputs)
    require(provenance["git_dirty"] is False and provenance["code_commit"] != "unavailable", "Clean committed reporter required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    periods.to_csv(args.output_dir / "period_comparison.csv", index=False)
    animals.to_csv(args.output_dir / "animal_comparison.csv", index=False)
    both = periods.groupby(KEY).expanded_better_than_global.all()
    report = {**provenance, "non_rescoring": True, "target_and_pair_denominators_unchanged": True,
              "frozen_pauses": new_manifest["frozen_pauses"], "measured_pauses": len(both),
              "animals": len(animals), "periods": len(periods), "dependent_pair_endpoints": len(new["pairs.csv"]),
              "positive_periods_matched": int(periods.matched_better_than_global.sum()),
              "positive_periods_expanded": int(periods.expanded_better_than_global.sum()),
              "pauses_positive_in_both_periods": int(both.sum()),
              "association_tested": False, "biological_calibration_complete": False, "goal_complete": False,
              "outputs_sha256": {name: file_sha256(args.output_dir / name) for name in ("period_comparison.csv", "animal_comparison.csv")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in provenance and k != "outputs_sha256"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
