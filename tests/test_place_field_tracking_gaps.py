from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import _frame_durations, _speed_cm_s
from hipporeplayimm.place_field_run_local_kinematics import (
    _durations_within_run_intervals,
    _mask_intervals_within_run_intervals,
    _speed_within_run_intervals,
)


def _dropout_times() -> np.ndarray:
    return np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2], dtype=float)


def _single_run() -> np.ndarray:
    return np.array([[0.0, 3.2]], dtype=float)


def test_run_local_durations_do_not_count_tracking_dropout_as_occupancy() -> None:
    durations = _durations_within_run_intervals(
        _dropout_times(),
        _single_run(),
        _frame_durations,
    )

    np.testing.assert_allclose(durations, np.full(6, 0.1), atol=1e-12)


def test_run_local_speed_does_not_bridge_tracking_dropout() -> None:
    times = _dropout_times()
    x = np.array([0.0, 1.0, 2.0, 1000.0, 1001.0, 1002.0], dtype=float)
    xy = np.column_stack([x, np.zeros_like(x)])

    speed = _speed_within_run_intervals(
        times,
        xy,
        _single_run(),
        _speed_cm_s,
    )

    np.testing.assert_allclose(speed, np.full(6, 10.0), atol=1e-12)


def test_position_training_intervals_split_at_tracking_dropout() -> None:
    times = _dropout_times()
    intervals = _mask_intervals_within_run_intervals(
        times,
        np.ones(times.shape, dtype=bool),
        _single_run(),
        _frame_durations,
    )

    np.testing.assert_allclose(
        intervals,
        np.array([[0.0, 0.3], [3.0, 3.2]], dtype=float),
        atol=1e-12,
    )
