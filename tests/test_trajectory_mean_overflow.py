from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def test_trajectory_mean_keeps_extreme_representable_convex_combination_finite() -> None:
    max_float = np.finfo(float).max
    metrics = trajectory_quality_metrics(
        np.zeros((2, 7), dtype=float),
        np.full(7, max_float, dtype=float),
    )

    assert metrics["trajectory_posterior_mean_path_length_cm"] == pytest.approx(0.0)
    assert metrics["trajectory_posterior_mean_displacement_cm"] == pytest.approx(0.0)
    assert metrics["trajectory_map_path_length_cm"] == pytest.approx(0.0)
    assert metrics["trajectory_mean_spread_cm"] == pytest.approx(0.0)
    assert metrics["trajectory_terminal_spread_cm"] == pytest.approx(0.0)
    assert all(np.isfinite(value) for value in metrics.values())
