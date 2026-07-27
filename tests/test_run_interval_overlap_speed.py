from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import _speed_cm_s
from hipporeplayimm.place_field_run_local_kinematics import (
    _speed_within_run_intervals,
)


def test_shared_run_endpoint_speed_is_independent_of_interval_order() -> None:
    times = np.array([0.0, 1.0, 2.0], dtype=float)
    xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [11.0, 0.0],
        ],
        dtype=float,
    )
    intervals = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )

    forward = _speed_within_run_intervals(times, xy, intervals, _speed_cm_s)
    reverse = _speed_within_run_intervals(times, xy, intervals[::-1], _speed_cm_s)

    np.testing.assert_allclose(forward, np.array([10.0, 10.0, 1.0]))
    np.testing.assert_allclose(reverse, forward)
