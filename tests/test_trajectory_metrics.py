import numpy as np
import pytest

from hipporeplayimm.models import LOG_ZERO
from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def test_trajectory_quality_metrics_rejects_empty_position_axis():
    with pytest.raises(ValueError, match="at least one time bin and one position bin"):
        trajectory_quality_metrics(
            np.empty((3, 0)),
            np.empty((0, 2)),
        )


def test_trajectory_quality_metrics_rejects_empty_coordinate_axis():
    with pytest.raises(ValueError, match="position_dim"):
        trajectory_quality_metrics(
            np.log(np.array([[1.0], [1.0]])),
            np.empty((1, 0)),
        )


def test_trajectory_quality_metrics_rejects_empty_time_axis():
    with pytest.raises(ValueError, match="at least one time bin and one position bin"):
        trajectory_quality_metrics(
            np.empty((0, 3)),
            np.zeros((3, 2)),
        )


def test_trajectory_quality_metrics_accepts_one_dimensional_bin_centers():
    metrics = trajectory_quality_metrics(
        np.log(np.array([[0.75, 0.25], [0.25, 0.75]])),
        np.array([0.0, 10.0]),
        times=np.array([0.0, 0.5]),
    )

    assert metrics["trajectory_time_bins"] == 2
    assert metrics["trajectory_posterior_mean_path_length_cm"] == pytest.approx(5.0)
    assert metrics["trajectory_posterior_mean_displacement_cm"] == pytest.approx(5.0)
    assert metrics["trajectory_posterior_mean_speed_cm_s"] == pytest.approx(10.0)
    assert metrics["trajectory_map_path_length_cm"] == pytest.approx(10.0)


def test_trajectory_quality_metrics_rejects_time_length_mismatch():
    with pytest.raises(ValueError, match="one timestamp"):
        trajectory_quality_metrics(
            np.log(np.array([[0.75, 0.25], [0.25, 0.75]])),
            np.array([0.0, 10.0]),
            times=np.array([0.0]),
        )


@pytest.mark.parametrize(
    ("times", "message"),
    [
        (np.array([0.0, np.nan]), "finite"),
        (np.array([0.5, 0.5]), "strictly increasing"),
        (np.array([1.0, 0.5]), "strictly increasing"),
    ],
)
def test_trajectory_quality_metrics_rejects_nonfinite_or_nonmonotone_times(times, message):
    with pytest.raises(ValueError, match=message):
        trajectory_quality_metrics(
            np.log(np.array([[0.75, 0.25], [0.25, 0.75]])),
            np.array([0.0, 10.0]),
            times=times,
        )


def test_trajectory_quality_metrics_entropy_ignores_impossible_bins():
    metrics = trajectory_quality_metrics(
        np.array(
            [
                [0.0, -np.inf, -np.inf],
                [-np.inf, 0.0, -np.inf],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=float,
        ),
    )

    assert metrics["trajectory_mean_entropy"] == pytest.approx(0.0)
    assert metrics["trajectory_terminal_entropy"] == pytest.approx(0.0)


def test_trajectory_quality_metrics_rejects_nonfinite_bin_centers():
    with pytest.raises(ValueError, match="bin_centers must contain finite values"):
        trajectory_quality_metrics(
            np.log(np.array([[0.5, 0.5]])),
            np.array([0.0, np.nan]),
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


def test_trajectory_quality_metrics_rejects_log_zero_sentinel_rows():
    with pytest.raises(ValueError, match="positive finite posterior mass"):
        trajectory_quality_metrics(
            np.full((2, 2), LOG_ZERO, dtype=float),
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
