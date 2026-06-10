from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_quality_audit import (
    ObservationCalibrationSelectionConfig,
    null_control_catalog,
    select_observation_calibration,
    write_result_quality_audit,
)


def test_select_observation_calibration_applies_behavior_and_recovery_gates() -> None:
    summary = pd.DataFrame(
        [
            {
                "setting": "bad_behavior",
                "median_posterior_mean_error_cm": 18.0,
                "simulation_recovery_accuracy": 0.90,
                "selection_used_real_evidence": False,
            },
            {
                "setting": "bad_recovery",
                "median_posterior_mean_error_cm": 12.0,
                "simulation_recovery_accuracy": 0.40,
                "selection_used_real_evidence": False,
            },
            {
                "setting": "real_selected",
                "median_posterior_mean_error_cm": 10.0,
                "simulation_recovery_accuracy": 0.80,
                "selection_used_real_evidence": True,
            },
            {
                "setting": "accepted",
                "median_posterior_mean_error_cm": 11.0,
                "simulation_recovery_accuracy": 0.75,
                "selection_used_real_evidence": False,
            },
        ]
    )

    selected = select_observation_calibration(
        summary,
        ObservationCalibrationSelectionConfig(
            max_behavior_error_cm=15.0,
            min_recovery_accuracy=0.60,
        ),
    )

    assert selected["setting"].tolist() == ["accepted"]
    assert bool(selected["selection_gate_passed"].iloc[0])


def test_select_observation_calibration_parses_string_bool_gate_columns() -> None:
    summary = pd.DataFrame(
        [
            {
                "setting": "accepted",
                "median_posterior_mean_error_cm": 10.0,
                "simulation_recovery_accuracy": 0.80,
                "selection_used_real_evidence": "False",
                "selection_passed_recovery_gate": "True",
                "candidate_support_quality_good": "True",
            },
            {
                "setting": "real_selected",
                "median_posterior_mean_error_cm": 9.0,
                "simulation_recovery_accuracy": 0.90,
                "selection_used_real_evidence": "True",
                "selection_passed_recovery_gate": "True",
                "candidate_support_quality_good": "False",
            },
            {
                "setting": "failed_recovery",
                "median_posterior_mean_error_cm": 8.0,
                "simulation_recovery_accuracy": 0.95,
                "selection_used_real_evidence": "False",
                "selection_passed_recovery_gate": "False",
                "candidate_support_quality_good": "False",
            },
        ]
    )

    selected = select_observation_calibration(
        summary,
        ObservationCalibrationSelectionConfig(
            max_behavior_error_cm=15.0,
            min_recovery_accuracy=0.60,
        ),
    )

    assert selected["setting"].tolist() == ["accepted"]
    assert bool(selected["selection_gate_passed"].iloc[0])


def test_null_control_catalog_lists_nonspatial_controls() -> None:
    catalog = null_control_catalog()

    names = set(catalog["null_control"])
    assert "circular spike-time shift" in names
    assert "clusterless mark-feature shuffle" in names
    assert "well-label shuffle" in names


def test_write_result_quality_audit_writes_core_outputs(tmp_path) -> None:
    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "random",
                "model_family": "nontrajectory",
                "log_evidence": -10.0,
                "n_time": 3,
                "n_spikes": 5,
                "runtime_s": 0.01,
                "error": "",
                "evidence_comparable": True,
                "evidence_support": "exact_full_grid",
                "relative_log_evidence": 0.0,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "stationary",
                "model_family": "nontrajectory",
                "log_evidence": -12.0,
                "n_time": 3,
                "n_spikes": 5,
                "runtime_s": 0.01,
                "error": "",
                "evidence_comparable": True,
                "evidence_support": "exact_full_grid",
                "relative_log_evidence": -2.0,
            },
        ]
    )

    dashboard = write_result_quality_audit(scores, tmp_path)

    assert dashboard.exists()
    assert (tmp_path / "evidence_margins.csv").exists()
    assert (tmp_path / "null_control_catalog.csv").exists()
    margins = pd.read_csv(tmp_path / "evidence_margins.csv")
    assert margins["best_model_by_evidence"].tolist() == ["random"]
    assert margins["evidence_margin_category"].tolist() == ["weak"]
