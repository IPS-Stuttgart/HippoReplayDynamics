"""Validate duration-aware momentum velocity-decay probabilities.

The active duration/occupancy scorer computes transition-specific momentum
multipliers from ``momentum_velocity_decay`` unless a physical decay time constant
is configured.  Lower-level candidate recursions already require velocity-decay
series to lie in ``[0, 1]``; keep the active duration-aware public scorer on the
same contract so accidental velocity amplification is rejected consistently.
"""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_momentum_velocity_decay_validation_patch_applied"
_ERROR_MESSAGE = "momentum_velocity_decay must be finite and in [0, 1]"


def _validate_decay(value: object) -> float:
    try:
        decay = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_MESSAGE) from exc
    if not np.isfinite(decay) or not 0.0 <= decay <= 1.0:
        raise ValueError(_ERROR_MESSAGE)
    return decay


def _uses_time_constant(config_or_decay: object) -> bool:
    if not hasattr(config_or_decay, "momentum_velocity_decay"):
        return False
    tau_s = float(getattr(config_or_decay, "momentum_velocity_decay_tau_s", 0.0))
    if not np.isfinite(tau_s) or tau_s < 0.0:
        # Let the wrapped scorer keep responsibility for the exact tau error text.
        return False
    return tau_s > 0.0


def _wrap_decay_series_function(function):
    if getattr(function, _PATCHED_FLAG, False):
        return function

    @wraps(function)
    def duration_adjusted_decays(config_or_decay, durations, reference_dt):
        if _uses_time_constant(config_or_decay):
            return function(config_or_decay, durations, reference_dt)
        if hasattr(config_or_decay, "momentum_velocity_decay"):
            _validate_decay(getattr(config_or_decay, "momentum_velocity_decay"))
        else:
            _validate_decay(config_or_decay)
        return function(config_or_decay, durations, reference_dt)

    setattr(duration_adjusted_decays, _PATCHED_FLAG, True)
    setattr(duration_adjusted_decays, "__hipporeplayimm_original__", function)
    return duration_adjusted_decays


def apply_momentum_velocity_decay_validation_patch() -> None:
    """Install validation in active and legacy duration-aware momentum helpers."""

    from . import duration_dynamics, duration_occupancy, state_space_sparse_momentum

    duration_occupancy._duration_adjusted_decays = _wrap_decay_series_function(
        duration_occupancy._duration_adjusted_decays,
    )
    # ``_duration_adjusted_decays_from_config`` delegates to
    # ``_duration_adjusted_decays`` for the tau_s == 0 path, so wrapping the
    # shared helper is sufficient for the active scorer.
    duration_dynamics._decays = _wrap_decay_series_function(duration_dynamics._decays)
    state_space_sparse_momentum._duration_adjusted_decays = _wrap_decay_series_function(
        state_space_sparse_momentum._duration_adjusted_decays,
    )


__all__ = ["apply_momentum_velocity_decay_validation_patch"]
