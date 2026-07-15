from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_quality_audit import (
    ObservationCalibrationSelectionConfig,
    select_observation_calibration,
)


def test_observation_calibration_accepts_arbitrary_precision_integer_flags() -> None:
    true_flag = 10**400
    summary = pd.DataFrame(
        {
            "setting": ["accepted", "real_selected"],
            "median_posterior_mean_error_cm": [10.0, 9.0],
            "simulation_recovery_accuracy": [0.80, 0.90],
            "selection_used_real_evidence": pd.Series([0, true_flag], dtype=object),
            "selection_passed_recovery_gate": pd.Series(
                [true_flag, true_flag],
                dtype=object,
            ),
            "candidate_support_quality_good": pd.Series(
                [true_flag, true_flag],
                dtype=object,
            ),
        }
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
