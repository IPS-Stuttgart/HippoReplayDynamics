from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_quality_audit import (
    ObservationCalibrationSelectionConfig,
    select_observation_calibration,
)


def test_select_observation_calibration_unwraps_singleton_gate_flags() -> None:
    summary = pd.DataFrame(
        [
            {
                "setting": "accepted",
                "median_posterior_mean_error_cm": 10.0,
                "simulation_recovery_accuracy": 0.80,
                "selection_used_real_evidence": np.array([False]),
                "selection_passed_recovery_gate": np.array([b"True"]),
                "candidate_support_quality_good": [[1]],
            },
            {
                "setting": "real_selected",
                "median_posterior_mean_error_cm": 9.0,
                "simulation_recovery_accuracy": 0.90,
                "selection_used_real_evidence": [np.bool_(True)],
                "selection_passed_recovery_gate": True,
                "candidate_support_quality_good": False,
            },
            {
                "setting": "failed_recovery",
                "median_posterior_mean_error_cm": 8.0,
                "simulation_recovery_accuracy": 0.95,
                "selection_used_real_evidence": False,
                "selection_passed_recovery_gate": (np.array([False]),),
                "candidate_support_quality_good": True,
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
