from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def test_trajectory_metrics_preserve_sub_epsilon_scale_invariance() -> None:
    scale = np.finfo(float).eps / 16.0

    metrics = trajectory_quality_metrics(
        np.array([[0.0, -np.inf], [-np.inf, 0.0]], dtype=float),
        np.array([0.0, scale], dtype=float),
        times=np.array([0.0, scale], dtype=float),
    )

    assert metrics["trajectory_posterior_mean_linearity"] == pytest.approx(1.0)
    assert metrics["trajectory_posterior_mean_speed_cm_s"] == pytest.approx(1.0)
    assert metrics["trajectory_direction_consistency"] == pytest.approx(1.0)


def test_trajectory_metrics_reject_unrepresentable_speed_ratio() -> None:
    with pytest.raises(ValueError, match="trajectory metric ratio exceeds floating-point range"):
        trajectory_quality_metrics(
            np.array([[0.0, -np.inf], [-np.inf, 0.0]], dtype=float),
            np.array([0.0, 1.0e308], dtype=float),
            times=np.array([0.0, 1.0e-308], dtype=float),
        )
