from __future__ import annotations

import numpy as np

from scripts.linearize_olafsdottir_ztrack import speed_from_linear_position


def test_linearized_speed_uses_nonuniform_timestamp_coordinates() -> None:
    times = np.array([0.0, 1.0, 3.0])
    linear = times**2
    valid = np.ones(times.shape, dtype=bool)

    speed = speed_from_linear_position(times, linear, valid)

    np.testing.assert_allclose(speed, np.array([1.0, 2.0, 4.0]))
