"""Validate PyRecEst goal-posterior diagnostics before reporting them."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_pyrecest_goal_diagnostics_validation_patch_applied"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_pyrecest_goal_diagnostics_validation_patch() -> None:
    """Install a guard for malformed PyRecEst goal posterior weights."""

    from . import pyrecest_models

    current = pyrecest_models._goal_diagnostics
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
    def goal_diagnostics(filter_: Any, goals: np.ndarray) -> dict[str, float | int]:
        try:
            goal_weights = np.asarray(filter_.get_goal_posterior_weights(goals), dtype=float)
        except (TypeError, ValueError):
            return {}

        goal_array = np.asarray(goals, dtype=float)
        if goal_array.ndim != 2 or goal_array.shape[0] == 0:
            return {}
        if goal_weights.ndim != 1 or goal_weights.shape != (goal_array.shape[0],):
            return {}
        if not np.all(np.isfinite(goal_weights)) or np.any(goal_weights < 0.0):
            return {}
        total_weight = float(goal_weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            return {}

        probabilities = goal_weights / total_weight
        idx = int(np.argmax(probabilities))
        return {
            "pyrecest_most_likely_goal_index": idx,
            "pyrecest_most_likely_goal_x": float(goal_array[idx, 0]),
            "pyrecest_most_likely_goal_y": (
                float(goal_array[idx, 1]) if goal_array.shape[1] > 1 else 0.0
            ),
            "pyrecest_most_likely_goal_probability": float(probabilities[idx]),
        }

    setattr(goal_diagnostics, _PATCHED_FLAG, True)
    setattr(goal_diagnostics, _ORIGINAL_ATTR, current)
    pyrecest_models._goal_diagnostics = goal_diagnostics


__all__ = ["apply_pyrecest_goal_diagnostics_validation_patch"]
