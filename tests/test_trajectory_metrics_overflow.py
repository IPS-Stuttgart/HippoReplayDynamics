from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


@pytest.mark.parametrize(
    ("trajectory_log_posterior", "bin_centers", "times", "message"),
    [
        (
            np.array([[10**400]], dtype=object),
            np.array([0.0]),
            None,
            "trajectory_log_posterior must contain numeric real values",
        ),
        (
            np.array([[0.0]]),
            np.array([10**400], dtype=object),
            None,
            "bin_centers must contain numeric real values",
        ),
        (
            np.array([[0.0], [0.0]]),
            np.array([0.0]),
            np.array([0, 10**400], dtype=object),
            "times must contain numeric real values",
        ),
    ],
)
def test_trajectory_quality_metrics_normalizes_float_conversion_overflow(
    trajectory_log_posterior: np.ndarray,
    bin_centers: np.ndarray,
    times: np.ndarray | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        trajectory_quality_metrics(
            trajectory_log_posterior,
            bin_centers,
            times=times,
        )
