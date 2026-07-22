from __future__ import annotations

import numpy as np

from hipporeplayimm.goal_state_space import _goal_drift_prediction


def test_goal_drift_preserves_representable_sub_epsilon_motion() -> None:
    distance = np.finfo(float).eps / 16.0

    midpoint = _goal_drift_prediction(
        np.array([0.0], dtype=float),
        np.array([distance], dtype=float),
        distance / 2.0,
    )
    reached = _goal_drift_prediction(
        np.array([0.0], dtype=float),
        np.array([distance], dtype=float),
        distance,
    )

    np.testing.assert_array_equal(midpoint, np.array([distance / 2.0]))
    np.testing.assert_array_equal(reached, np.array([distance]))
