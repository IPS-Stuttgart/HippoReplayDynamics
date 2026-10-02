"""Independently refit every cell in the fixed RUN-only diagnostic grid."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.verify_run_pair_glm_endpoint import global_predictions
    from scripts.verify_run_pair_rate_glm_stress import check, independent_residuals
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from verify_run_pair_glm_endpoint import global_predictions
    from verify_run_pair_rate_glm_stress import check, independent_residuals


def independent_score(counts, times, covariates, p):
    _, prediction = independent_residuals(counts, times, covariates, p, return_prediction=True)
    comparator = global_predictions(counts, times, p)
    gains = np.sum(xlogy(counts, prediction / comparator) - prediction + comparator, axis=0)
    return gains, prediction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    check(not args.output.exists(), "Do not overwrite independent verification")
    root = args.diagnostic_dir
    m = json.loads((root / "manifest.json").read_text())
    for name, digest in m["outputs_sha256"].items():
        check(file_sha256(root / name) == digest, f"Changed diagnostic output: {name}")
    for key, path in m["input_file_paths"].items():
        check(file_sha256(path) == m["input_file_sha256"][key], f"Changed diagnostic input: {key}")
    state = json.loads((root / "checkpoint_index.json").read_text())
    check(state["identity"] == {"code_commit": m["code_commit"], "inputs": m["input_file_sha256"]} and
          state["tasks"] == m["checkpoints_sha256"], "Checkpoint identities differ")
    for key, digest in state["tasks"].items():
        check(file_sha256(root / "checkpoints" / (key + ".json")) == digest, "Changed numerical checkpoint")
    p = json.loads(Path(m["input_file_paths"]["source_rate_protocol"]).read_text())
    diagnostic = json.loads(Path(m["input_file_paths"]["diagnostic_protocol"]).read_text())
    source_path = Path(m["input_file_paths"]["measurement_manifest"])
    sm = json.loads(source_path.read_text())
    for name in ("banks.csv", "pauses.csv"):
        check(file_sha256(source_path.parent / name) == sm["outputs_sha256"][name], "Changed source inventory")
    banks = pd.read_csv(source_path.parent / "banks.csv")
    source_pauses = pd.read_csv(source_path.parent / "pauses.csv").set_index(["animal", "session", "pause_id"])
    inventory = pd.read_csv(root / "inventory.csv").set_index(["animal", "session", "pause_id"])
    periods = pd.read_csv(root / "period_quality.csv", float_precision="round_trip")
    cells = pd.read_csv(root / "cell_quality.csv", float_precision="round_trip")
    summary = pd.read_csv(root / "candidate_summary.csv", float_precision="round_trip")
    grid = {}
    for spacing, penalty, mode in product(diagnostic["position_knot_cm"], diagnostic["l2_penalties"], diagnostic["interaction_modes"]):
        name = f"knot{spacing:g}_l2{penalty:g}_{mode}"
        check(name not in grid, "Duplicate model specification")
        grid[name] = {**p, "glm_position_knot_cm": spacing, "glm_l2_penalty": penalty,
                     "glm_spatial_direction_interaction": mode == "full", "glm_spatial_theta_interaction": mode == "full"}
    keys = ["animal", "session", "pause_id", "period", "candidate"]
    check(not periods.duplicated(keys).any() and not cells.duplicated(keys + ["unit_id"]).any(), "Duplicate diagnostic identity")
    check(inventory.index.is_unique and source_pauses.index.is_unique and
          set(inventory.index) == set(source_pauses.index) and len(inventory) == m["frozen_pauses"], "Frozen pause denominator changed")
    check(m["candidate_selected"] is m["association_fit"] is m["full_procedure_calibrated"] is m["goal_complete"] is False,
          "Diagnostic used to authorize inference")
    check(not cells.association_fit.any() and not cells.independent_biological_subject.any(), "Cell-level inference flags")
    check(not summary.selected_for_endpoint.any() and not summary.biological_inference.any(), "Candidate selection occurred")
    period_lookup, cell_lookup = periods.set_index(keys), cells.set_index(keys + ["unit_id"])
    check(len(summary) == len(grid) == m["diagnostic_candidates"] and set(summary.candidate) == set(grid), "Missing grid candidates")
    verified_cells, verified_periods, maximum_error = 0, 0, 0.
    for row in banks.itertuples(index=False):
        key = (row.animal, row.session, row.pause_id)
        check(file_sha256(row.bank_path) == row.bank_sha256 == inventory.loc[key, "bank_sha256"], "Changed source RUN bank")
        theta_ok = bool(source_pauses.loc[key, "theta_run_support_screen_passed"])
        check(inventory.loc[key, "status"] == ("included" if theta_ok else "frozen_theta_screen_failed"), "Source exclusion changed")
        if not theta_ok:
            check(not ((periods.session == row.session) & (periods.pause_id == row.pause_id)).any(), "Theta failure bypassed")
            continue
        with np.load(row.bank_path, allow_pickle=False) as bank:
            ids = bank["unit_ids"][:diagnostic["cells_per_pause"]]
            check(len(ids) == diagnostic["cells_per_pause"] and len(np.unique(ids)) == len(ids), "Changed frozen cells")
            for period in ("pre", "post"):
                phase = bank[f"{period}_theta_phase_rad"]
                valid = np.isfinite(phase).all(axis=1)
                times, counts = bank[f"{period}_time_s"][valid], bank[f"{period}_counts"][valid, :len(ids)]
                covariates = {name: bank[f"{period}_{column}"][valid] for name, column in
                    (("position", "position_cm"), ("direction", "direction_rad"), ("speed", "speed_cm_s"), ("theta", "theta_phase_rad"))}
                for name, model in grid.items():
                    selected = cell_lookup.loc[[( *key, period, name, int(unit)) for unit in ids]]
                    recorded = period_lookup.loc[(*key, period, name)]
                    check((selected.status == "complete").all() and recorded.status == "complete", "Incomplete real diagnostic fit")
                    check((selected.observed_bins == len(times)).all() and recorded.predicted_bins == recorded.total_bins == len(times), "RUN bins discarded")
                    np.testing.assert_array_equal(selected.observed_spikes, counts.sum(axis=0))
                    gains, predictions = independent_score(counts, times, covariates, model)
                    np.testing.assert_allclose(gains, selected.heldout_log_score_gain, rtol=1e-3, atol=2e-5)
                    np.testing.assert_allclose(gains.sum(), recorded.heldout_poisson_improvement_over_global, rtol=1e-3, atol=2e-5)
                    np.testing.assert_allclose(predictions.mean(axis=0), selected.predicted_mean_count, rtol=1e-3, atol=1e-8)
                    maximum_error = max(maximum_error, float(np.max(np.abs(gains - selected.heldout_log_score_gain.to_numpy()))))
                    verified_cells += len(ids)
                    verified_periods += 1
                print(f"VERIFIED independent RUN-only grid {row.animal} {row.pause_id} {period}", flush=True)
    check(verified_cells == len(cells) and verified_periods == len(periods) == len(state["tasks"]), "Incomplete refit accounting")
    for name, data in periods.groupby("candidate"):
        row = summary.set_index("candidate").loc[name]
        check(row.frozen_periods == row.complete_periods == len(data) and
              row.positive_periods == int((data.heldout_poisson_improvement_over_global > 0).sum()), "Candidate summary counts disagree")
        np.testing.assert_allclose(row.median_period_log_score_gain, data.heldout_poisson_improvement_over_global.median(), rtol=1e-12)
    provenance = build_script_provenance(input_paths={"diagnostic_manifest": root / "manifest.json"})
    check(provenance["git_dirty"] is False and provenance["code_commit"] != "unavailable", "Clean committed verifier required")
    result = {**provenance, "verified": True, "independently_refitted_cells": verified_cells,
        "independently_refitted_period_models": verified_periods, "maximum_absolute_cell_score_difference": maximum_error,
        "scope": "Every fixed eight-cell period/model refitted with independent SciPy B-splines and direct Poisson loss; all checkpoint identities, frozen exclusions and grid counts reconciled. Not full-population model adequacy, estimator selection, coordination calibration or biological inference.",
        "undefined_support_metric": "Producer labels are a single placeholder stratum for this prediction-only audit. Ignore unseen_stratum_fraction; it does not measure spatial support. Source theta masks and physical bin completeness are verified separately.",
        "association_verified": False, "full_procedure_calibrated": False, "goal_complete": False}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"COMPLETE independently refitted cells={verified_cells}; association_verified=False", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
