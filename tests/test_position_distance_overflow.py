from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import position_validation


def test_position_decoding_distance_stays_finite_for_large_coordinates() -> None:
    left = np.array([1e200, 1e200])
    right = np.zeros(2)

    distance = position_validation._distance(left, right)

    assert np.isfinite(distance)
    assert distance == pytest.approx(np.hypot(1e200, 1e200))
