from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from hipporeplayimm.encoding import _frame_durations
from hipporeplayimm.place_field_run_local_kinematics import (
    _durations_split_at_run_boundaries,
    _durations_within_run_intervals,
)


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        (
            np.array([0.0, 0.1, 0.6], dtype=float),
            np.array([0.1, 0.5, 0.5], dtype=float),
        ),
        (
            np.array([0.0, 0.5, 0.6], dtype=float),
            np.array([0.5, 0.1, 0.1], dtype=float),
        ),
    ],
)
@pytest.mark.parametrize(
    "duration_helper",
    [_durations_within_run_intervals, _durations_split_at_run_boundaries],
)
def test_shared_run_endpoint_duration_prefers_real_successor_and_is_order_independent(
    times: np.ndarray,
    expected: np.ndarray,
    duration_helper: Callable[..., np.ndarray],
) -> None:
    intervals = np.array(
        [
            [times[0], times[1]],
            [times[1], times[2]],
        ],
        dtype=float,
    )

    forward = duration_helper(times, intervals, _frame_durations)
    reverse = duration_helper(times, intervals[::-1], _frame_durations)

    np.testing.assert_allclose(forward, expected)
    np.testing.assert_allclose(reverse, expected)


@pytest.mark.parametrize(
    "duration_helper",
    [_durations_within_run_intervals, _durations_split_at_run_boundaries],
)
def test_overlapping_terminal_duration_fallback_is_order_independent(
    duration_helper: Callable[..., np.ndarray],
) -> None:
    times = np.array([0.0, 0.1, 0.6], dtype=float)
    intervals = np.array(
        [
            [times[0], times[2]],
            [times[1], times[2]],
        ],
        dtype=float,
    )

    forward = duration_helper(times, intervals, _frame_durations)
    reverse = duration_helper(times, intervals[::-1], _frame_durations)

    np.testing.assert_allclose(forward, np.array([0.1, 0.5, 0.3]))
    np.testing.assert_allclose(reverse, forward)
