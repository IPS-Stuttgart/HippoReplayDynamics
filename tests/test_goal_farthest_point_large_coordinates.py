from __future__ import annotations

import numpy as np

from hipporeplayimm.goal_state_space import _farthest_point_subset


def test_goal_farthest_subset_is_stable_for_large_finite_coordinates() -> None:
    points = (1e200 + np.arange(40, dtype=float) * 1e190)[:, None]

    with np.errstate(over="raise", invalid="raise"):
        subset = _farthest_point_subset(points, max_points=2)

    np.testing.assert_array_equal(subset, points[[0, -1]])
