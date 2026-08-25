"""Validate state-space occupancy support inputs.

State-space scoring derives an active-bin mask from per-bin occupancy seconds and
``valid_occupancy_threshold_s``.  The core helper previously delegated directly
to ``float(min_occupancy_s)`` and ``np.asarray(..., dtype=float)``, so booleans,
string numerals, array-shaped threshold values, and negative occupancy durations
could be silently coerced or treated as unsupported bins.  Reject those ambiguous
inputs before active support is computed.
"""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import numpy as np

_PATCHED_FLAG = "_state_space_occupancy_threshold_validation_patch_applied"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)


def _unwrap_scalar(value: Any, name: str) -> Any:
    """Recursively unwrap zero-dimensional object containers around one scalar."""

    current = value
    seen_arrays: set[int] = set()
    while True:
        try:
            raw = np.asarray(current)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a numeric scalar") from exc
        if raw.ndim != 0:
            raise TypeError(f"{name} must be a numeric scalar")
        if raw.dtype != object:
            try:
                return raw.item()
            except ValueError as exc:
                raise TypeError(f"{name} must be a numeric scalar") from exc

        if isinstance(current, np.ndarray):
            marker = id(current)
            if marker in seen_arrays:
                raise TypeError(f"{name} must be a numeric scalar")
            seen_arrays.add(marker)
        try:
            item = raw.item()
        except ValueError as exc:
            raise TypeError(f"{name} must be a numeric scalar") from exc
        if item is current:
            if isinstance(current, np.ndarray):
                raise TypeError(f"{name} must be a numeric scalar")
            return current
        current = item


def _validated_numeric_scalar(name: str, value: Any) -> Any:
    item = _unwrap_scalar(value, name)
    if isinstance(item, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not boolean")
    if isinstance(item, _STRING_TYPES):
        raise TypeError(f"{name} must be a numeric scalar, not string")
    return item


def _is_string_scalar(value: Any) -> bool:
    try:
        return isinstance(_unwrap_scalar(value, "value"), _STRING_TYPES)
    except TypeError:
        return False


def _reject_numeric_scalar_type(name: str, value: Any) -> None:
    _validated_numeric_scalar(name, value)


def _validated_occupancy_threshold(value: Any) -> float:
    item = _validated_numeric_scalar("min_occupancy_s", value)
    try:
        threshold = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("min_occupancy_s must be finite and nonnegative") from exc
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("min_occupancy_s must be finite and nonnegative")
    return threshold


def _validate_occupancy_seconds(occupancy_s: Any) -> None:
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

    numeric_source: Any = occupancy_s
    if raw.dtype == object:
        unwrapped: list[Any] = []
        for value in raw.reshape(-1):
            try:
                item = _unwrap_scalar(value, "occupancy_s")
            except TypeError as exc:
                raise TypeError("occupancy_s must contain numeric occupancy seconds") from exc
            if isinstance(item, (bool, np.bool_)):
                raise TypeError("occupancy_s must contain numeric seconds, not boolean values")
            if isinstance(item, _STRING_TYPES):
                raise TypeError("occupancy_s must contain numeric seconds, not string values")
            if isinstance(item, (complex, np.complexfloating)):
                raise TypeError("occupancy_s must contain real numeric seconds")
            unwrapped.append(item)
        numeric_source = np.asarray(unwrapped, dtype=object).reshape(raw.shape)

    try:
        numeric = np.asarray(numeric_source, dtype=float)
    except OverflowError as exc:
        raise ValueError("occupancy_s must contain finite occupancy seconds") from exc
    except (TypeError, ValueError) as exc:
        raise TypeError("occupancy_s must contain real numeric occupancy seconds") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("occupancy_s must contain finite occupancy seconds")
    if np.any(numeric < 0.0):
        raise ValueError("occupancy_s must contain nonnegative occupancy seconds")


def _sync_aliases(previous: Any, replacement: Any) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name != "hipporeplayimm" and not module_name.startswith("hipporeplayimm."):
            continue
        if getattr(module, "_valid_bin_mask_from_occupancy", None) is previous:
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
        _validate_occupancy_seconds(occupancy_s)
        threshold = _validated_occupancy_threshold(min_occupancy_s)
        return current(occupancy_s, threshold, n_bins)

    setattr(valid_bin_mask_from_occupancy, _PATCHED_FLAG, True)
    setattr(valid_bin_mask_from_occupancy, "__hipporeplayimm_original__", current)
    state_space_utils._valid_bin_mask_from_occupancy = valid_bin_mask_from_occupancy
    setattr(state_space_utils, _PATCHED_FLAG, True)
    _sync_aliases(current, valid_bin_mask_from_occupancy)


__all__ = ["apply_state_space_occupancy_threshold_validation_patch"]