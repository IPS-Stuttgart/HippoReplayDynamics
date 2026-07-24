from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import _speed_cm_s


def test_speed_uses_nonuniform_timestamp_coordinates() -> None:
    times = np.array([0.0, 1.0, 3.0])
    xy = np.column_stack((times**2, np.zeros_like(times)))

    speed = _speed_cm_s(times, xy)

    # The middle derivative of x=t^2 on this nonuniform grid is exactly 2.
    # Dividing gradient(x) by gradient(t) instead would incorrectly return 3.
    np.testing.assert_allclose(speed, np.array([1.0, 2.0, 4.0]))
