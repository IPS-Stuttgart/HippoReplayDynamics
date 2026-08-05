import warnings

import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def test_trajectory_mean_spread_avoids_reduction_overflow() -> None:
    log_posterior = np.log(np.full((3, 2), 0.5, dtype=float))
    bin_centers = np.array([-8.9e307, 8.9e307], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = trajectory_quality_metrics(log_posterior, bin_centers)

    assert metrics["trajectory_mean_spread_cm"] == pytest.approx(8.9e307)
    assert metrics["trajectory_terminal_spread_cm"] == pytest.approx(8.9e307)
