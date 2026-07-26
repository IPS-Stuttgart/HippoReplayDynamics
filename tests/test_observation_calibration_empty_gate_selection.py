from __future__ import annotations

import pandas as pd

import hipporeplayimm
from hipporeplayimm import result_quality_audit
from hipporeplayimm.result_quality_audit import (
    ObservationCalibrationSelectionConfig,
    select_observation_calibration,
)


def test_observation_calibration_returns_no_selection_when_every_gate_fails() -> None:
    summary = pd.DataFrame(
        [
            {
                "setting": "bad_behavior",
                "median_posterior_mean_error_cm": 20.0,
                "simulation_recovery_accuracy": 0.90,
                "selection_used_real_evidence": False,
            },
            {
                "setting": "bad_recovery",
                "median_posterior_mean_error_cm": 10.0,
                "simulation_recovery_accuracy": 0.40,
                "selection_used_real_evidence": False,
            },
            {
                "setting": "real_evidence_selected",
                "median_posterior_mean_error_cm": 10.0,
                "simulation_recovery_accuracy": 0.90,
                "selection_used_real_evidence": True,
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

    assert selected.empty
    assert "selection_gate_passed" in selected.columns
    assert "selection_rank" in selected.columns


def test_observation_calibration_gate_patch_is_idempotent() -> None:
    patched = result_quality_audit.select_observation_calibration

    hipporeplayimm.apply_runtime_patches()

    assert result_quality_audit.select_observation_calibration is patched
