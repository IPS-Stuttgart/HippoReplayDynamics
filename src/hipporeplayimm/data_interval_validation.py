"""Reject malformed dataset timing metadata before lossy float coercion.

The MATLAB/HDF5 loader exposes run, sleep, immobility, and REM intervals through
``data._as_intervals``. NumPy converts booleans to ``0``/``1`` and can discard
complex imaginary components while casting to float, allowing corrupted metadata
to look like valid epoch boundaries. Ripple-event rows have the same boolean-cast
hazard, so they are guarded before their legacy numeric conversion as well.
"""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import numpy as np

_PATCHED_FLAG = "_data_interval_validation_patch_applied"
_WRAPPER_MARK = "_data_interval_validation_wrapper"
_RIPPLE_WRAPPER_MARK = "_ripple_event_boolean_validation_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_MESSAGE = "Intervals must contain finite real numeric start and end times"
_RIPPLE_MESSAGE = "Ripple_Events must not contain boolean values"


def apply_data_interval_validation_patch() -> None:
    """Install strict scalar validation on shared dataset timing metadata."""

    from . import data

    current = data._as_intervals
    if bool(getattr(current, _WRAPPER_MARK, False)):
        original = getattr(current, _ORIGINAL_ATTR, None)
        if original is not None:
            _synchronize_interval_aliases(original, current)
    else:
        original = current

        @wraps(original)
        def as_intervals(value: Any) -> np.ndarray:
            _validate_interval_scalars(value)
            return original(value)

        setattr(as_intervals, _WRAPPER_MARK, True)
        setattr(as_intervals, _ORIGINAL_ATTR, original)
        data._as_intervals = as_intervals
        _synchronize_interval_aliases(original, as_intervals)

    current_ripple_validator = data._validate_ripple_event_times
    if not bool(getattr(current_ripple_validator, _RIPPLE_WRAPPER_MARK, False)):
        original_ripple_validator = current_ripple_validator

        @wraps(original_ripple_validator)
        def validate_ripple_event_times(events: np.ndarray) -> np.ndarray:
            if _contains_boolean_values(events):
                raise ValueError(_RIPPLE_MESSAGE)
            return original_ripple_validator(events)

        setattr(validate_ripple_event_times, _RIPPLE_WRAPPER_MARK, True)
        setattr(validate_ripple_event_times, _ORIGINAL_ATTR, original_ripple_validator)
        data._validate_ripple_event_times = validate_ripple_event_times

    setattr(data, _PATCHED_FLAG, True)


def _validate_interval_scalars(value: Any) -> None:
    """Reject malformed scalar leaves before NumPy's float conversion."""

    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_MESSAGE) from exc
    for item in raw.reshape(-1):
        _finite_real_interval_scalar(item)


def _finite_real_interval_scalar(value: Any) -> float:
    """Return one finite real scalar, recursively unwrapping 0-D arrays."""

    current = value
    seen_arrays: set[int] = set()
    while isinstance(current, np.ndarray):
        if current.ndim != 0:
            raise ValueError(_MESSAGE)
        identity = id(current)
        if identity in seen_arrays:
            raise ValueError(_MESSAGE)
        seen_arrays.add(identity)
        try:
            nested = current.item()
        except (TypeError, ValueError) as exc:
            raise ValueError(_MESSAGE) from exc
        if nested is current:
            raise ValueError(_MESSAGE)
        current = nested

    if isinstance(
        current,
        (
            bool,
            np.bool_,
            complex,
            np.complexfloating,
            str,
            bytes,
            bytearray,
            memoryview,
            np.str_,
            np.bytes_,
        ),
    ):
        raise ValueError(_MESSAGE)

    try:
        numeric = float(current)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_MESSAGE) from exc
    if not np.isfinite(numeric):
        raise ValueError(_MESSAGE)
    return numeric


def _contains_boolean_values(values: Any) -> bool:
    """Return whether an array-like value contains a boolean scalar."""

    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError, OverflowError):
        return False

    for item in raw.reshape(-1):
        current = item
        seen_arrays: set[int] = set()
        while isinstance(current, np.ndarray):
            if current.ndim != 0:
                break
            identity = id(current)
            if identity in seen_arrays:
                break
            seen_arrays.add(identity)
            try:
                nested = current.item()
            except (TypeError, ValueError):
                break
            if nested is current:
                break
            current = nested
        if isinstance(current, (bool, np.bool_)):
            return True
    return False


def _synchronize_interval_aliases(original: Any, replacement: Any) -> None:
    """Refresh package modules that imported ``_as_intervals`` by value."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_as_intervals", None) is original:
            module._as_intervals = replacement


__all__ = ["apply_data_interval_validation_patch"]
