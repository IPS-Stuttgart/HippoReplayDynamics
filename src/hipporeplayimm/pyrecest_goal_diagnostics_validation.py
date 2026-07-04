"""Validate PyRecEst goal-posterior diagnostics before reporting them."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_pyrecest_goal_diagnostics_validation_patch_applied"
_MODE_PATCHED_FLAG = "_pyrecest_mode_diagnostics_validation_patch_applied"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_pyrecest_goal_diagnostics_validation_patch() -> None:
    """Install guards for malformed PyRecEst posterior diagnostics."""

    from . import pyrecest_models

    current = pyrecest_models._goal_diagnostics
    if not getattr(current, _PATCHED_FLAG, False):

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

    current_mode = pyrecest_models._mode_diagnostics
    if getattr(current_mode, _MODE_PATCHED_FLAG, False):
        return

    @wraps(current_mode)
    def mode_diagnostics(filter_: Any) -> dict[str, float | str]:
        return _validated_mode_diagnostics(filter_)

    setattr(mode_diagnostics, _MODE_PATCHED_FLAG, True)
    setattr(mode_diagnostics, _ORIGINAL_ATTR, current_mode)
    pyrecest_models._mode_diagnostics = mode_diagnostics


def _validated_mode_diagnostics(filter_: Any) -> dict[str, float | str]:
    if not hasattr(filter_, "mode_probabilities"):
        return {}
    try:
        probabilities = np.asarray(filter_.mode_probabilities, dtype=float)
    except (TypeError, ValueError):
        return {}
    if probabilities.ndim != 1 or probabilities.size == 0:
        return {}
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        return {}
    total_probability = float(probabilities.sum())
    if not np.isfinite(total_probability) or total_probability <= 0.0:
        return {}
    normalized = probabilities / total_probability
    names = tuple(str(name) for name in getattr(filter_, "mode_names", ()))
    if len(names) != normalized.size:
        return {}
    diagnostics: dict[str, float | str] = {
        f"pyrecest_mode_{name}_probability": float(probability)
        for name, probability in zip(names, normalized, strict=True)
    }
    if hasattr(filter_, "most_likely_mode"):
        try:
            diagnostics["pyrecest_most_likely_mode"] = str(filter_.most_likely_mode())
        except (TypeError, ValueError):
            return {}
    if hasattr(filter_, "last_mode_transition_fraction"):
        try:
            transition_fraction = float(filter_.last_mode_transition_fraction)
        except (TypeError, ValueError):
            return diagnostics
        if np.isfinite(transition_fraction) and 0.0 <= transition_fraction <= 1.0:
            diagnostics["pyrecest_last_mode_transition_fraction"] = transition_fraction
    return diagnostics


__all__ = ["apply_pyrecest_goal_diagnostics_validation_patch"]
