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
