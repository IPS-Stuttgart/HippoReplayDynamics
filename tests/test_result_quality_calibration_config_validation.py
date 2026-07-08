from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_quality_audit import (
    ObservationCalibrationSelectionConfig,
    select_observation_calibration,
)


def _calibration_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "setting": "accepted",
                "median_posterior_mean_error_cm": 10.0,
                "simulation_recovery_accuracy": 0.80,
                "selection_used_real_evidence": False,
            }
        ]
    )


def test_select_observation_calibration_validates_top_k_for_empty_summary() -> None:
    with pytest.raises(TypeError, match="top_k"):
        select_observation_calibration(
            pd.DataFrame(),
            ObservationCalibrationSelectionConfig(top_k="2"),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("max_behavior_error_cm", True, TypeError),
        ("min_recovery_accuracy", np.bool_(False), TypeError),
        ("max_behavior_error_cm", "15.0", TypeError),
        ("min_recovery_accuracy", b"0.6", TypeError),
        ("max_behavior_error_cm", [15.0], TypeError),
        ("min_recovery_accuracy", np.array([0.6]), TypeError),
        ("max_behavior_error_cm", 15.0 + 0.0j, TypeError),
        ("min_recovery_accuracy", np.nan, ValueError),
        ("max_behavior_error_cm", np.inf, ValueError),
    ],
)
def test_select_observation_calibration_rejects_malformed_numeric_gates(
    field: str,
    value: object,
    expected_error: type[Exception],
) -> None:
    kwargs = {field: value}

    with pytest.raises(expected_error, match=field):
        select_observation_calibration(
            _calibration_summary(),
            ObservationCalibrationSelectionConfig(**kwargs),  # type: ignore[arg-type]
        )


def test_select_observation_calibration_accepts_numpy_real_gate_scalars() -> None:
    selected = select_observation_calibration(
        _calibration_summary(),
        ObservationCalibrationSelectionConfig(
            max_behavior_error_cm=np.float64(15.0),  # type: ignore[arg-type]
            min_recovery_accuracy=np.float32(0.6),  # type: ignore[arg-type]
        ),
    )

    assert selected["setting"].tolist() == ["accepted"]
