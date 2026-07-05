"""Validate state-space per-bin sigma helper inputs.

State-space diffusion and momentum models convert noise specified in
``cm/sqrt(s)`` to a per-bin standard deviation. Python booleans are numeric
subclasses, so ``float(True)`` silently becomes ``1.0`` unless the helpers reject
boolean values before coercion. Keep the guard at the shared helper boundary and
at the duration-aware scorer's private duplicate so direct and public import
surfaces enforce the same scalar contract.
"""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any

import numpy as np

_STATE_SPACE_UTILS_PATCHED_FLAG = "_state_space_per_bin_sigma_validation_patch_applied"
_DURATION_OCCUPANCY_PATCHED_FLAG = "_duration_occupancy_per_bin_sigma_validation_patch_applied"
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
    """Reject booleans, strings, and non-scalar values before float coercion."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if raw.ndim != 0:
        raise TypeError(f"{name} must be a numeric scalar")
    if _is_string_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not string")
    if isinstance(value, (bool, np.bool_)) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must be numeric, not boolean")
    if raw.dtype == object:
        try:
            item = raw.item()
        except ValueError as exc:
            raise TypeError(f"{name} must be a numeric scalar") from exc
        if isinstance(item, (bool, np.bool_)):
            raise TypeError(f"{name} must be numeric, not boolean")


def _validate_per_bin_sigma_inputs(sigma_cm_sqrt_s: Any, dt_s: Any) -> None:
    _reject_boolean_or_array_scalar("sigma_cm_sqrt_s", sigma_cm_sqrt_s)
    _reject_boolean_or_array_scalar("dt_s", dt_s)


def _patch_state_space_utils_sigma() -> None:
    from . import state_space_utils

    current = state_space_utils._per_bin_sigma
    if getattr(current, _STATE_SPACE_UTILS_PATCHED_FLAG, False):
        setattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, True)
        return

    @wraps(current)
    def per_bin_sigma(sigma_cm_sqrt_s, dt_s):
        _validate_per_bin_sigma_inputs(sigma_cm_sqrt_s, dt_s)
        return current(sigma_cm_sqrt_s, dt_s)

    setattr(per_bin_sigma, _STATE_SPACE_UTILS_PATCHED_FLAG, True)
    setattr(per_bin_sigma, "__hipporeplayimm_original__", current)
    state_space_utils._per_bin_sigma = per_bin_sigma
    setattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, True)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, "_per_bin_sigma", None) is current:
            module._per_bin_sigma = per_bin_sigma


def _patch_duration_occupancy_sigma() -> None:
    from . import duration_occupancy

    current = duration_occupancy._per_bin_sigma
    if getattr(current, _DURATION_OCCUPANCY_PATCHED_FLAG, False):
        setattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, True)
        return

    @wraps(current)
    def per_bin_sigma(sigma_cm_sqrt_s, dt_s):
        _validate_per_bin_sigma_inputs(sigma_cm_sqrt_s, dt_s)
        return current(sigma_cm_sqrt_s, dt_s)

    setattr(per_bin_sigma, _DURATION_OCCUPANCY_PATCHED_FLAG, True)
    setattr(per_bin_sigma, "__hipporeplayimm_original__", current)
    duration_occupancy._per_bin_sigma = per_bin_sigma
    setattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, True)


def apply_state_space_sigma_validation_patch() -> None:
    """Install idempotent validation for per-bin sigma conversion helpers."""

    _patch_state_space_utils_sigma()
    _patch_duration_occupancy_sigma()


__all__ = ["apply_state_space_sigma_validation_patch"]
