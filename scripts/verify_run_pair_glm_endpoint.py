"""Independently refit bounded GLM RUN endpoints; not a biological association."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.verify_run_pair_coordination_endpoint import independent_coordination
    from scripts.verify_run_pair_rate_glm_stress import check, independent_residuals
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from verify_run_pair_coordination_endpoint import independent_coordination
    from verify_run_pair_rate_glm_stress import check, independent_residuals


def global_predictions(counts, times, p):
    """Training-only comparator, independently reconstructing guarded folds."""
    blocks = np.floor(times / p["crossfit_time_block_s"]).astype(np.int64)
    prediction = np.full(counts.shape, np.nan)
    for fold in range(p["crossfit_folds"]):
        target = blocks % p["crossfit_folds"] == fold
        if not target.any():
            continue
        training = ~target
        for block in np.unique(blocks[target]):
            start = block * p["crossfit_time_block_s"] - p["crossfit_guard_s"]
            stop = (block + 1) * p["crossfit_time_block_s"] + p["crossfit_guard_s"]
            training &= ~((times >= start) & (times < stop))
        check(training.any(), "Missing guarded global-rate training")
        rate = (counts[training].sum(axis=0) + p["global_rate_prior_spikes"]) / (
            training.sum() * p["run_bin_s"] + p["global_rate_prior_exposure_s"])
        prediction[target] = np.maximum(rate * p["run_bin_s"], p["mean_count_floor"])
    check(np.isfinite(prediction).all(), "Incomplete independent global-rate predictions")
    return prediction


def independently_measure_period(bank, period, p):
    phase = np.asarray(bank[f"{period}_theta_phase_rad"])
    check(phase.ndim == 2 and phase.shape[1] > 0, "Missing native theta references")
    valid = np.isfinite(phase).all(axis=1)
    counts = bank[f"{period}_counts"][valid]
    times = bank[f"{period}_time_s"][valid]
    check(len(times) > 0 and np.all(np.diff(times) > 0), "Missing or backward RUN clock")
    check(np.allclose(bank[f"{period}_bin_duration_s"], p["run_bin_s"], rtol=0, atol=1e-12),
          "RUN bin width differs")
    covariates = {key: bank[f"{period}_{name}"][valid] for key, name in
                  (("position", "position_cm"), ("direction", "direction_rad"),
                   ("speed", "speed_cm_s"), ("theta", "theta_phase_rad"))}
    check(all(np.isfinite(value).all() for value in covariates.values()), "Missing nuisance covariates")
    check(np.all(covariates["speed"] > p["speed_edges_cm_s"][0]) and
          np.all(covariates["speed"] <= p["speed_edges_cm_s"][-1]), "RUN speed outside frozen bounds")
    residual, mean = independent_residuals(counts, times, covariates, p, return_prediction=True)
    matrix, opportunities = independent_coordination(residual, times, p)
    baseline = global_predictions(counts, times, p)
    improvement = float(np.sum(xlogy(counts, mean / baseline) - mean + baseline))
    return matrix, {"source_bins": len(phase), "usable_bins": len(times),
        "predicted_bins": len(times), "total_bins": len(times),
        "physical_lag_opportunities": opportunities,
        "heldout_poisson_improvement_over_global": improvement,
        "minimum_predicted_mean_count": float(mean.min()),
        "maximum_predicted_mean_count": float(mean.max())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    check(not args.output.exists(), "Do not overwrite a verification artifact")
    root = args.endpoint_dir
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["outputs_sha256"].items():
        check(file_sha256(root / name) == digest, f"Changed endpoint output {name}")
    for name, path in manifest["input_file_paths"].items():
        check(file_sha256(path) == manifest["input_file_sha256"][name], f"Changed endpoint input {name}")
    p = json.loads(Path(manifest["input_file_paths"]["protocol"]).read_text())
    check(p.get("rate_model_family") == "smooth_poisson_glm" and
          p.get("glm_speed_scaling") == "fixed_log_run_bounds", "Bounded GLM protocol required")
    source_path = Path(manifest["input_file_paths"]["measurement_manifest"])
    source = source_path.parent
    source_manifest = json.loads(source_path.read_text())
    verification = json.loads(Path(manifest["input_file_paths"]["measurement_verification"]).read_text())
    check(verification.get("verified") is True and
          verification["input_file_sha256"]["manifest"] == file_sha256(source_path),
          "Identity-matched source measurement verification required")
    for name, digest in source_manifest["outputs_sha256"].items():
        check(file_sha256(source / name) == digest, f"Changed source measurement {name}")
    banks = pd.read_csv(source / "banks.csv")
    source_pauses = pd.read_csv(source / "pauses.csv").set_index(["session", "pause_id"])
    pauses = pd.read_csv(root / "pauses.csv").set_index(["session", "pause_id"])
    pairs = pd.read_csv(root / "pairs.csv", float_precision="round_trip")
    quality = pd.read_csv(root / "period_quality.csv", float_precision="round_trip").set_index(
        ["session", "pause_id", "period"])
    check(source_pauses.index.is_unique and pauses.index.is_unique and quality.index.is_unique and
          not banks.duplicated(["session", "pause_id"]).any(), "Duplicate frozen identities")
    bank_keys = set(zip(banks.session, banks.pause_id, strict=True))
    check(set(source_pauses.index) == set(pauses.index) == bank_keys and
          len(banks) == manifest["frozen_pauses"], "Changed frozen pause denominator")
    check(not pairs.duplicated(["session", "pause_id", "unit_a", "unit_b"]).any(), "Duplicate pair endpoint")
    check(not pairs.independent_biological_subject.any() and not pairs.association_fit.any(), "Invalid claim flags")
    check(manifest["candidate_endpoint_measured"] == (len(pairs) > 0) and
          manifest["association_fit"] is False and manifest["biological_calibration_complete"] is False and
          manifest["goal_complete"] is False and manifest["inference_unit"] == "animal", "Invalid manifest claims")
    np.testing.assert_allclose(pairs.coordination_change, pairs.post_coordination - pairs.pre_coordination,
                               rtol=0, atol=1e-12)
    checked_pairs, checked_bins, checked_quality = 0, 0, set()
    period_audit, unavailable = [], []
    for row in banks.itertuples(index=False):
        key = (row.session, row.pause_id)
        check(row.animal == source_pauses.loc[key, "animal"] == pauses.loc[key, "animal"], "Animal changed")
        check(file_sha256(row.bank_path) == row.bank_sha256, "Changed source bank")
        selected = pairs[(pairs.session == row.session) & (pairs.pause_id == row.pause_id)]
        if not bool(source_pauses.loc[key, "theta_run_support_screen_passed"]):
            check(selected.empty and pauses.loc[key, "status"] == "frozen_theta_screen_failed", "Failed theta screen bypassed")
            continue
        with np.load(row.bank_path, allow_pickle=False) as bank:
            ids = bank["unit_ids"]
            check(len(ids) == len(np.unique(ids)), "Duplicate unit identity")
            a, b = np.triu_indices(len(ids), 1)
            if pauses.loc[key, "status"] == "endpoint_unavailable":
                check(selected.empty and pauses.loc[key, "pair_endpoints"] == 0, "Unavailable pause has endpoints")
                check(any(quality.loc[(*key, period), "status"] != "measured" for period in ("pre", "post")),
                      "Measured periods silently discarded")
                unavailable.append({"animal": row.animal, "session": row.session, "pause_id": row.pause_id})
                checked_quality.update((*key, period) for period in ("pre", "post"))
                continue
            check(pauses.loc[key, "status"] == "candidate_endpoint_measured" and
                  len(selected) == len(a) == pauses.loc[key, "pair_endpoints"], "Incomplete fixed pair family")
            selected = selected.set_index(["unit_a", "unit_b"]).loc[list(zip(ids[a], ids[b], strict=True))]
            for period in ("pre", "post"):
                check(quality.loc[(*key, period), "status"] == "measured", "Unmeasured period used")
                matrix, qc = independently_measure_period(bank, period, p)
                recorded = quality.loc[(*key, period)]
                for name in ("source_bins", "usable_bins", "predicted_bins", "total_bins", "physical_lag_opportunities"):
                    check(recorded[name] == qc[name], f"RUN support differs: {name}")
                np.testing.assert_allclose(recorded.heldout_poisson_improvement_over_global,
                    qc["heldout_poisson_improvement_over_global"], rtol=1e-3, atol=2e-5)
                np.testing.assert_allclose(selected[f"{period}_coordination"], matrix[a, b], rtol=1e-3, atol=2e-7)
                error = float(np.max(np.abs(selected[f"{period}_coordination"].to_numpy() - matrix[a, b])))
                period_audit.append({"animal": row.animal, "session": row.session, "pause_id": row.pause_id,
                                     "period": period, "max_absolute_endpoint_refit_difference": error, **qc})
                checked_bins += qc["usable_bins"]
                checked_quality.add((*key, period))
            checked_pairs += len(selected)
        print(f"VERIFIED bounded RUN endpoint {row.animal} {row.pause_id}: {len(selected)} dependent pairs", flush=True)
    check(checked_pairs == len(pairs) == manifest["dependent_pair_endpoints"] and
          len(period_audit) // 2 == manifest["measured_pauses"] and
          set(quality.index) == checked_quality, "Endpoint accounting incomplete")
    provenance = build_script_provenance(input_paths={"endpoint_manifest": root / "manifest.json"})
    check(provenance["git_dirty"] is False and provenance["code_commit"] != "unavailable", "Clean committed verifier required")
    result = {**provenance, "verified": True, "checked_pair_endpoints": checked_pairs,
        "checked_run_bins": checked_bins, "checked_frozen_pauses": len(banks),
        "measured_pauses": len(period_audit) // 2, "unavailable_pauses": unavailable,
        "period_audit": period_audit,
        "method": "Independent SciPy B-splines, direct Poisson optimization and FFT physical-lag endpoints",
        "scope": "Every available real RUN endpoint independently refitted; unavailable pauses retained but not numerically certified. Not replay validation, full null calibration or association.",
        "association_verified": False, "biological_calibration_verified": False, "goal_complete": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"COMPLETE independently refitted={checked_pairs} pairs; association_verified=False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
