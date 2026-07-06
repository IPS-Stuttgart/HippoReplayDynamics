"""Validate state-space occupancy support inputs.

State-space scoring derives an active-bin mask from per-bin occupancy seconds and
``valid_occupancy_threshold_s``.  The core helper previously delegated directly
to ``float(min_occupancy_s)`` and ``np.asarray(..., dtype=float)``, so booleans,
string numerals, and array-shaped threshold values could be silently coerced into
valid numeric masks.  Reject those ambiguous inputs before active support is
computed.
"""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import numpy as np

_PATCHED_FLAG = "_state_space_occupancy_threshold_validation_patch_applied"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)


def _is_string_scalar(value: Any) -> bool:
    if isinstance(value, _STRING_TYPES):
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
            return isinstance(raw.item(), _STRING_TYPES)
        except ValueError:
            return False
    return False


def _reject_numeric_scalar_type(name: str, value: Any) -> None:
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
        if isinstance(item, _STRING_TYPES):
            raise TypeError(f"{name} must be a numeric scalar, not string")


def _reject_occupancy_seconds_type(occupancy_s: Any) -> None:
    if occupancy_s is None:
        return
    try:
        raw = np.asarray(occupancy_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("occupancy_s must contain numeric occupancy seconds") from exc
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("occupancy_s must contain numeric seconds, not boolean values")
    if np.issubdtype(raw.dtype, np.str_) or np.issubdtype(raw.dtype, np.bytes_):
        raise TypeError("occupancy_s must contain numeric seconds, not string values")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError("occupancy_s must contain real numeric seconds")
    if raw.dtype == object:
        flat = raw.reshape(-1)
        if any(isinstance(value, (bool, np.bool_)) for value in flat):
            raise TypeError("occupancy_s must contain numeric seconds, not boolean values")
        if any(isinstance(value, _STRING_TYPES) for value in flat):
            raise TypeError("occupancy_s must contain numeric seconds, not string values")


def _sync_aliases(previous: Any, replacement: Any) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, "_valid_bin_mask_from_occupancy", None) is previous:
            module._valid_bin_mask_from_occupancy = replacement


def apply_state_space_occupancy_threshold_validation_patch() -> None:
    """Install idempotent validation for occupancy-derived active support."""

    from . import state_space_utils

    current = state_space_utils._valid_bin_mask_from_occupancy
    if getattr(current, _PATCHED_FLAG, False):
        original = getattr(current, "__hipporeplayimm_original__", None)
        if original is not None:
            _sync_aliases(original, current)
        setattr(state_space_utils, _PATCHED_FLAG, True)
        return

    @wraps(current)
    def valid_bin_mask_from_occupancy(occupancy_s, min_occupancy_s, n_bins):
        _reject_occupancy_seconds_type(occupancy_s)
        _reject_numeric_scalar_type("min_occupancy_s", min_occupancy_s)
        return current(occupancy_s, min_occupancy_s, n_bins)

    setattr(valid_bin_mask_from_occupancy, _PATCHED_FLAG, True)
    setattr(valid_bin_mask_from_occupancy, "__hipporeplayimm_original__", current)
    state_space_utils._valid_bin_mask_from_occupancy = valid_bin_mask_from_occupancy
    setattr(state_space_utils, _PATCHED_FLAG, True)
    _sync_aliases(current, valid_bin_mask_from_occupancy)


__all__ = ["apply_state_space_occupancy_threshold_validation_patch"]
