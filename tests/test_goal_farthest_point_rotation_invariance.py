from __future__ import annotations

import numpy as np

from hipporeplayimm.goal_state_space import _farthest_point_subset


def test_goal_farthest_subset_is_rotation_invariant() -> None:
    points = np.array(
        [
            [3.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -2.0],
        ]
    )
    angle = np.pi / 4.0
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    subset = _farthest_point_subset(points, max_points=3)
    rotated_subset = _farthest_point_subset(points @ rotation.T, max_points=3)

    np.testing.assert_allclose(
        rotated_subset,
        subset @ rotation.T,
        rtol=1e-14,
        atol=1e-14,
    )
