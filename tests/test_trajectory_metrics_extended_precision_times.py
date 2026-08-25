import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def test_trajectory_quality_metrics_preserves_extended_precision_time_differences():
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("np.longdouble has no precision beyond binary64 on this platform")

    base = np.longdouble(2) ** 60
    times = np.array([base, base + 1, base + 3], dtype=np.longdouble)
    narrowed_times = np.asarray(times, dtype=float)

    assert np.any(np.diff(narrowed_times) <= 0.0)

    metrics = trajectory_quality_metrics(
        np.array(
            [
                [0.0, -np.inf],
                [-np.inf, 0.0],
                [-np.inf, 0.0],
            ]
        ),
        np.array([0.0, 3.0]),
        times=times,
    )

    assert metrics["trajectory_posterior_mean_path_length_cm"] == pytest.approx(3.0)
    assert metrics["trajectory_posterior_mean_speed_cm_s"] == pytest.approx(1.0)
