"""Validate momentum candidate-prediction decay multipliers.

Momentum candidate augmentation uses duration-aware displacement multipliers
before nearest-bin lookup.  Non-finite values create NaN/Inf predicted positions
and can silently select arbitrary spatial bins through ``argmin``.  Keep the
guard close to the helper so direct lower-level imports follow the same
validation contract as package-level model scoring.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_momentum_prediction_decay_validation_patch_applied"


def _coerce_prediction_multiplier(name: str, value: object) -> float:
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    try:
        multiplier = float(arr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative scalar") from exc
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return multiplier


def apply_momentum_prediction_decay_validation_patch() -> None:
    """Install validation for state-space momentum candidate predictors."""

    from . import state_space_model

    current = state_space_model._transition_decay_at
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
    def transition_decay_at(values, transition_index: int, fallback):
        if values is None:
            return _coerce_prediction_multiplier("velocity_decay", fallback)

        arr = np.asarray(values)
        if arr.ndim == 0:
            if int(transition_index) == 0:
                return _coerce_prediction_multiplier("velocity_decays", arr)
            return _coerce_prediction_multiplier("velocity_decay", fallback)
        if arr.ndim != 1:
            raise ValueError("velocity_decays must be a finite nonnegative scalar or one-dimensional sequence")
        if int(transition_index) < 0 or int(transition_index) >= arr.size:
            return _coerce_prediction_multiplier("velocity_decay", fallback)
        return _coerce_prediction_multiplier("velocity_decays", arr[int(transition_index)])

    setattr(transition_decay_at, _PATCHED_FLAG, True)
    setattr(transition_decay_at, "__hipporeplayimm_original__", current)
    state_space_model._transition_decay_at = transition_decay_at


__all__ = ["apply_momentum_prediction_decay_validation_patch"]
