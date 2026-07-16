"""Validate state-space per-bin sigma helper inputs.

State-space diffusion and momentum models convert noise specified in
``cm/sqrt(s)`` to a per-bin standard deviation. Python booleans are numeric
subclasses, and NumPy complex scalars can be coerced to floats by discarding the
imaginary component, so validate scalar types before float conversion. Keep the
guard at the shared helper boundary and at the duration-aware scorer's private
duplicate so direct and public import surfaces enforce the same scalar contract.
"""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any, Callable

import numpy as np

_STATE_SPACE_UTILS_PATCHED_FLAG = "_state_space_per_bin_sigma_validation_patch_applied"
_DURATION_OCCUPANCY_PATCHED_FLAG = "_duration_occupancy_per_bin_sigma_validation_patch_applied"
_MODE_TRANSITION_PATCHED_FLAG = "_state_space_mode_transition_string_validation_patch_applied"
_PER_BIN_SIGMA_WRAPPER_VERSION = 2
_STRING_SCALAR_TYPES = (str, bytes, np.str_, np.bytes_)


def _is_string_scalar(value: Any) -> bool:
    if isinstance(value, _STRING_SCALAR_TYPES):
        return True
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if raw.ndim != 0:
        return False
    if np.issubdtype(raw.dtype, np.str_) or np.issubdtype(raw.dtype, np.bytes_):
        return True
    if raw.dtype == object:
        try:
            return isinstance(raw.item(), _STRING_SCALAR_TYPES)
        except ValueError:
            return False
    return False


def _reject_boolean_or_array_scalar(name: str, value: Any) -> None:
    """Reject booleans, strings, complex values, and non-scalar inputs."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if raw.ndim != 0:
        raise TypeError(f"{name} must be a numeric scalar")
    if _is_string_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not string")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"{name} must be real-valued, not complex")
    if isinstance(value, (bool, np.bool_)) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must be numeric, not boolean")
    if raw.dtype == object:
        try:
            item = raw.item()
        except ValueError as exc:
            raise TypeError(f"{name} must be a numeric scalar") from exc
        if isinstance(item, (complex, np.complexfloating)):
            raise TypeError(f"{name} must be real-valued, not complex")
        if isinstance(item, (bool, np.bool_)):
            raise TypeError(f"{name} must be numeric, not boolean")


def _validate_per_bin_sigma_inputs(sigma_cm_sqrt_s: Any, dt_s: Any) -> None:
    _reject_boolean_or_array_scalar("sigma_cm_sqrt_s", sigma_cm_sqrt_s)
    _reject_boolean_or_array_scalar("dt_s", dt_s)


def _validated_per_bin_sigma(
    base: Callable[[Any, Any], float],
    sigma_cm_sqrt_s: Any,
    dt_s: Any,
) -> float:
    """Call a per-bin sigma helper and reject derived floating-point overflow."""

    _validate_per_bin_sigma_inputs(sigma_cm_sqrt_s, dt_s)
    with np.errstate(over="ignore", invalid="ignore"):
        process_sigma = base(sigma_cm_sqrt_s, dt_s)
    if not np.isfinite(process_sigma):
        raise ValueError(
            "sigma_cm_sqrt_s and dt_s must produce a finite per-bin sigma"
        )
    return float(process_sigma)


def _patch_state_space_utils_sigma() -> None:
    from . import state_space_utils

    observed = state_space_utils._per_bin_sigma
    if getattr(observed, _STATE_SPACE_UTILS_PATCHED_FLAG, None) == _PER_BIN_SIGMA_WRAPPER_VERSION:
        setattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, True)
        return
    base = getattr(observed, "__hipporeplayimm_original__", observed)

    @wraps(base)
    def per_bin_sigma(sigma_cm_sqrt_s, dt_s):
        return _validated_per_bin_sigma(base, sigma_cm_sqrt_s, dt_s)

    setattr(per_bin_sigma, _STATE_SPACE_UTILS_PATCHED_FLAG, _PER_BIN_SIGMA_WRAPPER_VERSION)
    setattr(per_bin_sigma, "__hipporeplayimm_original__", base)
    state_space_utils._per_bin_sigma = per_bin_sigma
    setattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, True)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        alias = getattr(module, "_per_bin_sigma", None)
        if module_name.startswith("hipporeplayimm") and (alias is observed or alias is base):
            module._per_bin_sigma = per_bin_sigma


def _patch_duration_occupancy_sigma() -> None:
    from . import duration_occupancy

    observed = duration_occupancy._per_bin_sigma
    if getattr(observed, _DURATION_OCCUPANCY_PATCHED_FLAG, None) == _PER_BIN_SIGMA_WRAPPER_VERSION:
        setattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, True)
        return
    base = getattr(observed, "__hipporeplayimm_original__", observed)

    @wraps(base)
    def per_bin_sigma(sigma_cm_sqrt_s, dt_s):
        return _validated_per_bin_sigma(base, sigma_cm_sqrt_s, dt_s)

    setattr(per_bin_sigma, _DURATION_OCCUPANCY_PATCHED_FLAG, _PER_BIN_SIGMA_WRAPPER_VERSION)
    setattr(per_bin_sigma, "__hipporeplayimm_original__", base)
    duration_occupancy._per_bin_sigma = per_bin_sigma
    setattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, True)


def _patch_state_space_utils_mode_transition() -> None:
    from . import state_space_utils

    current = state_space_utils._mode_transition_matrix
    if getattr(current, _MODE_TRANSITION_PATCHED_FLAG, False):
        setattr(state_space_utils, _MODE_TRANSITION_PATCHED_FLAG, True)
        return

    @wraps(current)
    def mode_transition_matrix(n_modes, stickiness):
        _reject_boolean_or_array_scalar("mode_stickiness", stickiness)
        try:
            numeric_stickiness = float(np.asarray(stickiness).item())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("mode_stickiness must be in [0, 1]") from exc
        return current(n_modes, numeric_stickiness)

    setattr(mode_transition_matrix, _MODE_TRANSITION_PATCHED_FLAG, True)
    setattr(mode_transition_matrix, "__hipporeplayimm_original__", current)
    state_space_utils._mode_transition_matrix = mode_transition_matrix
    setattr(state_space_utils, _MODE_TRANSITION_PATCHED_FLAG, True)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, "_mode_transition_matrix", None) is current:
            module._mode_transition_matrix = mode_transition_matrix


def apply_state_space_sigma_validation_patch() -> None:
    """Install idempotent validation for state-space scalar conversion helpers."""

    _patch_state_space_utils_sigma()
    _patch_duration_occupancy_sigma()
    _patch_state_space_utils_mode_transition()


__all__ = ["apply_state_space_sigma_validation_patch"]
