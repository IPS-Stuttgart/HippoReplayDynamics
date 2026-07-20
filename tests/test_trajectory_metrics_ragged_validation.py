from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


@pytest.mark.parametrize(
    ("trajectory_log_posterior", "bin_centers", "times", "field"),
    [
        (
            [[0.0], [0.0, -1.0]],
            [0.0],
            None,
            "trajectory_log_posterior",
        ),
        (
            np.log(np.array([[0.5, 0.5]])),
            [[0.0], [1.0, 2.0]],
            None,
            "bin_centers",
        ),
        (
            np.log(np.array([[1.0], [1.0]])),
            [0.0],
            [[0.0], [0.5, 1.0]],
            "times",
        ),
    ],
)
def test_trajectory_quality_metrics_rejects_ragged_inputs_with_field_context(
    trajectory_log_posterior,
    bin_centers,
    times,
    field,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must contain numeric real values"):
        trajectory_quality_metrics(
            trajectory_log_posterior,
            bin_centers,
            times=times,
        )
