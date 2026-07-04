"""Validate mark-matrix candidates before numeric coercion.

Spike marks are analog waveform or amplitude features.  Complex values with a
nonzero imaginary component, and boolean values that are likely logical metadata
rather than marks, must be rejected before the legacy coercion path can silently
convert them to real-valued features.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_mark_complex_validation_patch_applied"
_PATCH_WRAPPER_ATTR = "_mark_complex_validation_wrapper"


def _complex_has_zero_imaginary(values: np.ndarray) -> bool:
    imaginary = np.imag(values)
    return bool(np.all(np.isfinite(imaginary)) and np.allclose(imaginary, 0.0, rtol=0.0, atol=0.0))


def _contains_boolean_values(values: Any) -> bool:
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, np.ndarray):
        raw = values
        if raw.size == 0:
            return False
        if np.issubdtype(raw.dtype, np.bool_):
            return True
        if raw.dtype == object:
            return any(_contains_boolean_values(value) for value in raw.reshape(-1))
        return False
    if isinstance(values, Mapping):
        return any(_contains_boolean_values(value) for value in values.values())
    if isinstance(values, (str, bytes, bytearray)):
        return False
    if isinstance(values, Iterable):
        return any(_contains_boolean_values(value) for value in values)
    return False


def _is_mark_complex_validation_wrapper(func: object) -> bool:
    return bool(getattr(func, _PATCH_WRAPPER_ATTR, False))


def apply_mark_complex_validation_patch() -> None:
    """Install value-type validation for mark-matrix candidates and direct mark likelihood calls."""

    from . import clusterless, data

    current_coerce_mark_matrix = data._coerce_mark_matrix
    if not _is_mark_complex_validation_wrapper(current_coerce_mark_matrix):
        original_coerce_mark_matrix = current_coerce_mark_matrix

        @wraps(original_coerce_mark_matrix)
        def coerce_mark_matrix(value, *, spike_count: int, spike_times: np.ndarray):
            if _contains_boolean_values(value):
                return None
            arr = np.asarray(value)
            if arr.dtype.kind == "c":
                if not _complex_has_zero_imaginary(arr):
                    return None
                value = np.real(arr)
            return original_coerce_mark_matrix(value, spike_count=spike_count, spike_times=spike_times)

        setattr(coerce_mark_matrix, _PATCH_WRAPPER_ATTR, True)
        data._coerce_mark_matrix = coerce_mark_matrix

    current_coerce_marks = clusterless.ClusterlessMarkEncoding._coerce_marks
    if not _is_mark_complex_validation_wrapper(current_coerce_marks):
        original_coerce_marks = current_coerce_marks

        @wraps(original_coerce_marks)
        def coerce_marks(self, marks):
            if _contains_boolean_values(marks):
                raise ValueError("marks must contain numeric mark features, not boolean values")
            values = np.asarray(marks)
            if values.dtype.kind == "c":
                if not _complex_has_zero_imaginary(values):
                    raise ValueError("marks must not contain complex values with nonzero imaginary components")
                marks = np.real(values)
            coerced = original_coerce_marks(self, marks)
            if not np.all(np.isfinite(coerced)):
                raise ValueError("marks must contain finite mark features")
            return coerced

        setattr(coerce_marks, _PATCH_WRAPPER_ATTR, True)
        clusterless.ClusterlessMarkEncoding._coerce_marks = coerce_marks

    setattr(data, _PATCHED_FLAG, True)
    setattr(clusterless, _PATCHED_FLAG, True)


__all__ = ["apply_mark_complex_validation_patch"]
