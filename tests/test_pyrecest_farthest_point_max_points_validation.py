from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.pyrecest_models import _farthest_point_subset


@pytest.mark.parametrize(
    "max_points",
    [
        True,
        np.array(True, dtype=object),
        2.5,
        np.array([2]),
        0,
    ],
)
def test_pyrecest_farthest_point_subset_rejects_invalid_max_points(max_points: object) -> None:
    points = np.arange(12, dtype=float).reshape(6, 2)

    with pytest.raises(ValueError, match="max_points must be a positive integer"):
        _farthest_point_subset(points, max_points=max_points)


def test_pyrecest_farthest_point_subset_accepts_numpy_integer_max_points() -> None:
    points = np.arange(12, dtype=float).reshape(6, 2)

    subset = _farthest_point_subset(points, max_points=np.int64(3))

    assert subset.shape == (3, 2)
