from __future__ import annotations

import numpy as np

import hipporeplayimm
from hipporeplayimm.pyrecest_models import _goal_diagnostics


class _GoalWeightFilter:
    def __init__(self, weights: object) -> None:
        self._weights = weights

    def get_goal_posterior_weights(self, goals: np.ndarray) -> object:
        return self._weights


def test_pyrecest_goal_diagnostics_normalizes_valid_weights() -> None:
    hipporeplayimm.apply_runtime_patches()
    goals = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)

    diagnostics = _goal_diagnostics(_GoalWeightFilter(np.array([2.0, 6.0], dtype=float)), goals)

    assert diagnostics["pyrecest_most_likely_goal_index"] == 1
    assert diagnostics["pyrecest_most_likely_goal_x"] == 2.0
    assert diagnostics["pyrecest_most_likely_goal_y"] == 3.0
    assert diagnostics["pyrecest_most_likely_goal_probability"] == 0.75


def test_pyrecest_goal_diagnostics_skips_invalid_weights() -> None:
    hipporeplayimm.apply_runtime_patches()
    goals = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    invalid_weights = [
        np.array([np.nan, 1.0], dtype=float),
        np.array([0.0, -1.0], dtype=float),
        np.array([0.0, 0.0], dtype=float),
        np.array([[0.5, 0.5]], dtype=float),
        np.array([1.0], dtype=float),
    ]

    for weights in invalid_weights:
        assert _goal_diagnostics(_GoalWeightFilter(weights), goals) == {}
