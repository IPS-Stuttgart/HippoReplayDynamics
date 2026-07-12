import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def test_trajectory_spread_ignores_impossible_bins_with_extreme_coordinates() -> None:
    metrics = trajectory_quality_metrics(
        np.array(
            [
                [0.0, -np.inf],
                [0.0, -np.inf],
            ],
            dtype=float,
        ),
        np.array([0.0, np.finfo(float).max], dtype=float),
    )

    assert metrics["trajectory_mean_spread_cm"] == pytest.approx(0.0)
    assert metrics["trajectory_terminal_spread_cm"] == pytest.approx(0.0)


def test_trajectory_spread_rejects_positive_mass_beyond_float_range() -> None:
    with pytest.raises(ValueError, match="posterior spread exceeds floating-point range"):
        trajectory_quality_metrics(
            np.log(np.array([[0.5, 0.5]], dtype=float)),
            np.array(
                [-np.finfo(float).max, np.finfo(float).max],
                dtype=float,
            ),
        )


def test_trajectory_metrics_reject_unrepresentable_path_geometry() -> None:
    with pytest.raises(ValueError, match="trajectory path geometry exceeds floating-point range"):
        trajectory_quality_metrics(
            np.array([[0.0, -np.inf], [-np.inf, 0.0]], dtype=float),
            np.array([-np.finfo(float).max, np.finfo(float).max], dtype=float),
        )


def test_trajectory_metrics_keep_representable_large_step_finite() -> None:
    metrics = trajectory_quality_metrics(
        np.array([[0.0, -np.inf], [-np.inf, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [1.0e308, 1.0e308]], dtype=float),
    )

    expected = np.sqrt(2.0) * 1.0e308
    assert metrics["trajectory_posterior_mean_path_length_cm"] == pytest.approx(expected)
    assert metrics["trajectory_posterior_mean_displacement_cm"] == pytest.approx(expected)
    assert metrics["trajectory_map_path_length_cm"] == pytest.approx(expected)
    assert metrics["trajectory_direction_consistency"] == pytest.approx(1.0)
    assert all(np.isfinite(value) for value in metrics.values())


def test_trajectory_metrics_reject_unrepresentable_timestamp_difference() -> None:
    with pytest.raises(ValueError, match="timestamp differences exceed floating-point range"):
        trajectory_quality_metrics(
            np.array([[0.0], [0.0]], dtype=float),
            np.array([0.0], dtype=float),
            times=np.array([-np.finfo(float).max, np.finfo(float).max], dtype=float),
        )


def test_trajectory_metrics_reject_unrepresentable_total_duration() -> None:
    with pytest.raises(ValueError, match="total trajectory duration exceeds floating-point range"):
        trajectory_quality_metrics(
            np.array([[0.0], [0.0], [0.0]], dtype=float),
            np.array([0.0], dtype=float),
            times=np.array([-np.finfo(float).max, 0.0, np.finfo(float).max], dtype=float),
        )
