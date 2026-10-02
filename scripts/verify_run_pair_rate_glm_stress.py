"""Independently audit a development GLM stress; no biological association."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import iv
from scipy.stats import t

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.verify_run_pair_coordination_endpoint import independent_coordination
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from verify_run_pair_coordination_endpoint import independent_coordination


def check(value, message):
    if not value:
        raise ValueError(message)


def features(training, target, p):
    """SciPy B-splines, not the producer's sklearn transformer or tensor builder."""
    matrices = []
    for data in (training, target):
        axes = []
        for k in (0, 1):
            step, degree = p["glm_position_knot_cm"], p["glm_spline_degree"]
            low = np.floor(training["position"][:, k].min() / step) * step
            high = np.ceil(training["position"][:, k].max() / step) * step
            high = max(high, low + step)
            knots = np.arange(low - degree * step, high + (degree + .5) * step, step)
            values = np.clip(data["position"][:, k], low, high)
            axes.append(sparse.csr_matrix(BSpline.design_matrix(values, knots, degree)))
        spatial = sparse.hstack([axes[0][:, a].multiply(axes[1][:, b])
            for a in range(axes[0].shape[1]) for b in range(axes[1].shape[1])], format="csr")
        direction = np.array([fun(h * data["direction"])
            for h in range(1, p["glm_direction_harmonics"] + 1) for fun in (np.sin, np.cos)]).T
        theta = np.array([fun(h * data["theta"][:, k]) for k in range(data["theta"].shape[1])
            for h in range(1, p["glm_theta_harmonics"] + 1) for fun in (np.sin, np.cos)]).T
        scale = max(float(np.std(np.log(training["speed"]))), .1)
        speed = (np.log(data["speed"]) - np.mean(np.log(training["speed"]))) / scale
        blocks = [spatial, sparse.csr_matrix(direction), sparse.csr_matrix(theta),
                  sparse.csr_matrix(np.array([speed**i for i in range(1, p["glm_speed_degree"] + 1)]).T)]
        if p["glm_spatial_direction_interaction"]:
            blocks.extend(spatial.multiply(direction[:, i, None]) for i in (0, 1))
        if p["glm_spatial_theta_interaction"]:
            blocks.extend(spatial.multiply(theta[:, 2 * p["glm_theta_harmonics"] * k + i, None])
                          for k in range(data["theta"].shape[1]) for i in (0, 1))
        matrices.append(sparse.hstack(blocks, format="csr"))
    return matrices


def independent_residuals(counts, times, covariates, p):
    """Direct weighted Poisson objective; no producer rate fitting or cached design."""
    prediction = np.full(counts.shape, np.nan)
    block = np.floor(times / p["crossfit_time_block_s"]).astype(int)
    for fold in range(p["crossfit_folds"]):
        target = block % p["crossfit_folds"] == fold
        if not target.any():
            continue
        train = ~target
        for b in np.unique(block[target]):
            start = b * p["crossfit_time_block_s"] - p["crossfit_guard_s"]
            stop = (b + 1) * p["crossfit_time_block_s"] + p["crossfit_guard_s"]
            train &= ~((times >= start) & (times < stop))
        check(train.any(), "Independent audit lacks guarded training")
        x, z = features({k: v[train] for k, v in covariates.items()},
                        {k: v[target] for k, v in covariates.items()}, p)
        x = sparse.vstack((x, sparse.csr_matrix(np.asarray(x.mean(axis=0)))), format="csr")
        weight = np.r_[np.ones(train.sum()), p["global_rate_prior_exposure_s"] / p["run_bin_s"]]
        penalty = p["glm_l2_penalty"] / weight.sum()
        weight /= weight.sum()

        def fit_cell(k):
            y = np.r_[counts[train, k], p["global_rate_prior_spikes"] * p["run_bin_s"] /
                      p["global_rate_prior_exposure_s"]]
            initial = np.r_[np.zeros(x.shape[1]), np.log(weight @ y)]
            def objective(coefficient):
                eta = np.asarray(x @ coefficient[:-1]).ravel() + coefficient[-1]
                mu = np.exp(eta)
                loss = weight @ (mu - y * eta) + penalty * (coefficient[:-1] @ coefficient[:-1]) / 2
                error = weight * (mu - y)
                gradient = np.r_[np.asarray(x.T @ error).ravel() + penalty * coefficient[:-1], error.sum()]
                return loss, gradient
            result = minimize(objective, initial, method="L-BFGS-B", jac=True,
                options={"maxiter": p["glm_max_iter"], "maxls": 50, "gtol": 1e-9,
                         "ftol": 64 * np.finfo(float).eps})
            check(result.success, f"Independent Poisson fit failed: {result.message}")
            return np.exp(np.asarray(z @ result.x[:-1]).ravel() + result.x[-1])
        with ThreadPoolExecutor(max_workers=p["glm_workers"]) as pool:
            prediction[target] = np.column_stack(list(pool.map(fit_cell, range(counts.shape[1]))))
    check(np.isfinite(prediction).all(), "Incomplete independently fitted predictions")
    prediction = np.maximum(prediction, p["mean_count_floor"])
    return (counts - prediction) / np.sqrt(prediction)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-dir", type=Path, required=True)
    parser.add_argument("--previous-stress-dir", type=Path, help="Immutable original stress for paired revision accounting")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.stress_dir
    manifest = json.loads((root / "manifest.json").read_text())
    check(manifest["rate_model_family"] == "smooth_poisson_glm", "GLM development stress required")
    for name, digest in manifest["outputs_sha256"].items():
        check(file_sha256(root / name) == digest, f"Changed stress output {name}")
    for name, digest in manifest["input_file_sha256"].items():
        check(file_sha256(manifest["input_file_paths"][name]) == digest, f"Changed stress input {name}")
    p = json.loads(Path(manifest["input_file_paths"]["estimator_protocol"]).read_text())
    stress = json.loads(Path(manifest["input_file_paths"]["stress_protocol"]).read_text())
    frame, summary, inventory, fits = [pd.read_csv(root / f"{name}.csv")
        for name in ("replicates", "pair_stress_summary", "inventory", "fit_quality")]
    identity = ["animal", "session", "pause_id", "unit_a", "unit_b"]
    check(not frame.duplicated(identity + ["replicate"]).any(), "Duplicate replica identity")
    check(not summary.duplicated(identity).any(), "Duplicate summary identity")
    check(not inventory.animal.duplicated().any(), "More than one selected pause per animal")
    check(len(inventory) == manifest["selected_animals"], "Animal denominator differs")
    check(len(frame) == len(summary) * stress["replicates"], "Incomplete replica denominator")
    check(len(summary) == len(inventory) * stress["cells_per_pause"] * (stress["cells_per_pause"] - 1) // 2,
          "Missing fixed pair family")
    check(len(fits) == len(inventory) * stress["replicates"] * 2 and
          not fits.duplicated(["animal", "session", "pause_id", "replicate", "period"]).any(), "Incomplete fit inventory")
    check((fits.predicted_bins == fits.total_bins).all(), "Unfavorable RUN bins were discarded")
    for row in fits.itertuples(index=False):
        fold = json.loads(row.fold_diagnostics)
        check(len(fold) == p["crossfit_folds"] and all(f["status"] == "predicted" and
              f["all_cells_converged"] and f["max_fit_iterations"] < p["glm_max_iter"] for f in fold),
              "Incomplete or nonconverged fit")
    np.testing.assert_allclose(frame.discrepancy, frame.fitted_change - frame.oracle_change, rtol=1e-12, atol=1e-12)
    critical = t.ppf(1 - .05 / (2 * len(summary)), stress["replicates"] - 1)
    computed = frame.groupby(identity, sort=True).discrepancy.agg(["mean", "std", "count"])
    saved = summary.set_index(identity).loc[computed.index]
    check((computed["count"] == stress["replicates"]).all(), "Incomplete per-pair draws")
    np.testing.assert_allclose(saved.mean_discrepancy, computed["mean"], rtol=1e-12, atol=1e-12)
    half = critical * computed["std"] / np.sqrt(stress["replicates"])
    np.testing.assert_allclose(saved.simultaneous_mc_interval_low, computed["mean"] - half, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(saved.simultaneous_mc_interval_high, computed["mean"] + half, rtol=1e-12, atol=1e-12)
    flags = (computed["mean"] - half > 0) | (computed["mean"] + half < 0)
    check(np.array_equal(saved.systematic_nuisance_bias_detected, flags) and
          int(flags.sum()) == manifest["biased_animal_pairs"], "Bias decision differs")
    previous = None
    if args.previous_stress_dir is not None:
        old = json.loads((args.previous_stress_dir / "manifest.json").read_text())
        for name, digest in old["outputs_sha256"].items():
            check(file_sha256(args.previous_stress_dir / name) == digest, f"Changed previous output {name}")
        for name, digest in old["input_file_sha256"].items():
            check(file_sha256(old["input_file_paths"][name]) == digest, f"Changed previous input {name}")
        old_protocol = json.loads(Path(old["input_file_paths"]["stress_protocol"]).read_text())
        check(all(old_protocol[k] == stress[k] for k in
                  ("seed", "replicates", "cells_per_pause", "base_rate_hz", "theta_concentration",
                   "post_even_cell_phase_shift_rad")), "Generator changed between development runs")
        old_inventory = pd.read_csv(args.previous_stress_dir / "inventory.csv")
        pd.testing.assert_frame_equal(inventory, old_inventory)
        old_frame = pd.read_csv(args.previous_stress_dir / "replicates.csv").set_index(identity + ["replicate"])
        current = frame.set_index(identity + ["replicate"])
        check(old_frame.index.is_unique and set(old_frame.index) == set(current.index), "Changed replica cohort")
        np.testing.assert_allclose(current.oracle_change, old_frame.loc[current.index].oracle_change,
                                   rtol=1e-12, atol=1e-12)
        previous = {"previous_biased_pairs": old["biased_animal_pairs"],
                    "amended_biased_pairs": manifest["biased_animal_pairs"],
                    "all_oracle_endpoints_unchanged": True, "cohort_and_generator_unchanged": True}
    checks = []
    for row in inventory.itertuples(index=False):
        check(file_sha256(row.bank_path) == row.bank_sha256, "Changed frozen bank")
        with np.load(row.bank_path, allow_pickle=False) as bank:
            ids = bank["unit_ids"][:stress["cells_per_pause"]]
            refs = bank["unit_theta_reference_index"][:len(ids)]
            salt = int.from_bytes(hashlib.sha256(f"{row.session}:{row.pause_id}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(np.random.SeedSequence([stress["seed"], salt]))
            scores = {}
            for period in ("pre", "post"):
                valid = np.isfinite(bank[f"{period}_theta_phase_rad"]).all(axis=1)
                c = {key: bank[f"{period}_{name}"][valid] for key, name in
                     (("position", "position_cm"), ("direction", "direction_rad"),
                      ("speed", "speed_cm_s"), ("theta", "theta_phase_rad"))}
                times = bank[f"{period}_time_s"][valid]
                preferences = np.arange(len(ids)) * 2 * np.pi / len(ids)
                if period == "post":
                    preferences[::2] += stress["post_even_cell_phase_shift_rad"]
                mu = p["run_bin_s"] * stress["base_rate_hz"] * np.exp(stress["theta_concentration"] *
                    np.cos(c["theta"][:, refs] - preferences)) / iv(0, stress["theta_concentration"])
                counts = rng.poisson(mu)
                fitted = independent_residuals(counts, times, c, p)
                a, opportunities = independent_coordination(fitted, times, p)
                b, oracle_opportunities = independent_coordination((counts - mu) / np.sqrt(mu), times, p)
                check(opportunities == oracle_opportunities and opportunities > 0, "Physical-lag support changed")
                scores[period] = (a, b)
            selected = frame[(frame.animal == row.animal) & (frame.session == row.session) &
                             (frame.pause_id == row.pause_id) & (frame.replicate == 0)]
            check(len(selected) == len(ids) * (len(ids) - 1) // 2, "Missing first-replica fixed pairs")
            errors = []
            for pair in selected.itertuples(index=False):
                a, b = int(np.flatnonzero(ids == pair.unit_a)[0]), int(np.flatnonzero(ids == pair.unit_b)[0])
                actual = scores["post"][0][a, b] - scores["pre"][0][a, b]
                oracle = scores["post"][1][a, b] - scores["pre"][1][a, b]
                np.testing.assert_allclose(oracle, pair.oracle_change, rtol=1e-10, atol=1e-12)
                # Independent optimizers have finite stopping error; this is a numerical, not biological bound.
                np.testing.assert_allclose(actual, pair.fitted_change, rtol=1e-3, atol=2e-7)
                errors.append(abs(actual - pair.fitted_change))
            checks.append({"animal": row.animal, "first_replica_pairs": len(selected),
                           "max_absolute_endpoint_refit_difference": max(errors)})
            print(f"VERIFIED first-replica independent GLM/FFT {row.animal}", flush=True)
    inputs = {"stress_manifest": root / "manifest.json"}
    if args.previous_stress_dir is not None:
        inputs["previous_stress_manifest"] = args.previous_stress_dir / "manifest.json"
    provenance = build_script_provenance(input_paths=inputs)
    check(provenance["git_dirty"] is False, "Clean committed verifier checkout required")
    result = {**provenance, "verified": True, "replica_rows_accounted": len(frame),
        "summary_rows_accounted": len(summary), "fit_rows_accounted": len(fits), "first_replicates": checks,
        "paired_revision_comparison": previous,
        "scope": "All hashes, fit completeness, fixed pair counts, discrepancy arithmetic and Monte Carlo intervals; independently regenerated/fitted only first replica per animal using SciPy B-splines, direct Poisson loss and FFT. Not all random draws, a biological model, or full null calibration.",
        "association_fit": False, "full_procedure_calibrated": False, "goal_complete": False}
    check(not args.output.exists(), "Verification artifact already exists; do not overwrite")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
