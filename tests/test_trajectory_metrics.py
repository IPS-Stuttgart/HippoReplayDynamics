import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def test_trajectory_quality_metrics_rejects_empty_position_axis():
    with pytest.raises(ValueError, match="at least one time bin and one position bin"):
        trajectory_quality_metrics(
            np.empty((3, 0)),
            np.empty((0, 2)),
        )


def test_trajectory_quality_metrics_rejects_empty_time_axis():
    with pytest.raises(ValueError, match="at least one time bin and one position bin"):
        trajectory_quality_metrics(
            np.empty((0, 3)),
            np.zeros((3, 2)),
        )


def test_trajectory_quality_metrics_rejects_nan_or_positive_inf():
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="cannot contain NaN or \+inf"):
        trajectory_quality_metrics(
            np.array([[0.0, np.nan]]),
            centers,
        )

    with pytest.raises(ValueError, match="cannot contain NaN or \+inf"):
        trajectory_quality_metrics(
            np.array([[0.0, np.inf]]),
            centers,
        )


def test_trajectory_quality_metrics_rejects_rows_without_finite_mass():
    with pytest.raises(ValueError, match="finite posterior mass"):
        trajectory_quality_metrics(
            np.array([[0.0, -np.inf], [-np.inf, -np.inf]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        )


def test_trajectory_quality_metrics_accepts_single_time_bin():
    metrics = trajectory_quality_metrics(
        np.log(np.array([[0.2, 0.8]])),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    )

    assert metrics["trajectory_time_bins"] == 1
    assert metrics["trajectory_map_path_length_cm"] == 0.0
    assert metrics["trajectory_posterior_mean_path_length_cm"] == 0.0
