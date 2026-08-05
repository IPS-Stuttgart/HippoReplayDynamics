from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(float).nmant,
    reason="platform longdouble does not provide additional timestamp precision",
)
def test_trajectory_quality_metrics_preserves_extended_precision_timestamp_differences() -> None:
    base = np.longdouble(2) ** 60
    times = np.asarray(
        [
            base,
            base + np.longdouble(1),
            base + np.longdouble(3),
        ],
        dtype=np.longdouble,
    )
    assert np.array_equal(np.diff(times), np.asarray([1.0, 2.0], dtype=np.longdouble))
    assert np.array_equal(np.diff(times.astype(float)), np.asarray([0.0, 0.0]))

    log_posterior = np.asarray(
        [
            [0.0, -np.inf, -np.inf],
            [-np.inf, 0.0, -np.inf],
            [-np.inf, -np.inf, 0.0],
        ],
        dtype=float,
    )
    metrics = trajectory_quality_metrics(
        log_posterior,
        np.asarray([0.0, 2.0, 6.0], dtype=float),
        times=times,
    )

    assert metrics["trajectory_posterior_mean_path_length_cm"] == pytest.approx(6.0)
    assert metrics["trajectory_posterior_mean_speed_cm_s"] == pytest.approx(2.0)
    assert metrics["trajectory_map_path_length_cm"] == pytest.approx(6.0)
