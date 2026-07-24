from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_quality_audit import (
    ObservationCalibrationSelectionConfig,
    select_observation_calibration,
)


def test_select_observation_calibration_decodes_byte_backed_gate_flags() -> None:
    summary = pd.DataFrame(
        [
            {
                "setting": "accepted",
                "median_posterior_mean_error_cm": 10.0,
                "simulation_recovery_accuracy": 0.80,
                "selection_used_real_evidence": b"False",
                "selection_passed_recovery_gate": np.bytes_("True"),
                "candidate_support_quality_good": memoryview(b"True"),
            },
            {
                "setting": "real_selected",
                "median_posterior_mean_error_cm": 9.0,
                "simulation_recovery_accuracy": 0.90,
                "selection_used_real_evidence": bytearray(b"True"),
                "selection_passed_recovery_gate": b"True",
                "candidate_support_quality_good": b"False",
            },
            {
                "setting": "failed_recovery",
                "median_posterior_mean_error_cm": 8.0,
                "simulation_recovery_accuracy": 0.95,
                "selection_used_real_evidence": np.bytes_("False"),
                "selection_passed_recovery_gate": memoryview(b"False"),
                "candidate_support_quality_good": b"False",
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
