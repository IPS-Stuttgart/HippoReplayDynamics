import numpy as np

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def test_trajectory_direction_consistency_is_bounded_by_one():
    step = np.array([-4.14985471e-83, -3.28534772e-87])
    centers = np.stack([np.zeros(2), step, 2.0 * step])
    log_posterior = np.full((3, 3), -np.inf)
    np.fill_diagonal(log_posterior, 0.0)

    metrics = trajectory_quality_metrics(log_posterior, centers)

    assert metrics["trajectory_direction_consistency"] == 1.0
