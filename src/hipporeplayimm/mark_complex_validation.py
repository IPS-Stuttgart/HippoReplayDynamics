"""Validate mark-matrix candidates before numeric coercion.

Spike marks are analog waveform or amplitude features. Complex values with a
nonzero imaginary component, boolean values that are likely logical metadata,
and text-valued observations must be rejected before legacy coercion paths can
silently convert them to real-valued features. Ambiguous square mark matrices are
reoriented when an embedded spike-time column uniquely identifies the spike axis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCHED_FLAG = "_mark_complex_validation_patch_applied"
_DATA_WRAPPER_ATTR = "_mark_complex_validation_wrapper"
# Backward-compatible private alias retained for runtime-patch regression tests
# and any callers that still inspect the original wrapper marker name.
_PATCH_WRAPPER_ATTR = _DATA_WRAPPER_ATTR
_CLUSTERLESS_WRAPPER_ATTR = "_mark_complex_validation_clusterless_wrapper"


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


def _contains_text_values(values: Any) -> bool:
    if isinstance(values, (str, bytes, bytearray)):
        return True
    if isinstance(values, np.ndarray):
        raw = values
        if raw.size == 0:
            return False
        if raw.dtype.kind in {"S", "U"}:
            return True
        if raw.dtype == object:
            return any(_contains_text_values(value) for value in raw.reshape(-1))
        return False
    if isinstance(values, Mapping):
        return any(_contains_text_values(value) for value in values.values())
    if isinstance(values, Iterable):
        return any(_contains_text_values(value) for value in values)
    return False


def _real_observation_marks_or_raise(values: Any) -> Any:
    """Reject lossy observation coercions and normalize zero-imaginary values."""

    if _contains_boolean_values(values):
        raise ValueError("marks must contain real-valued numeric mark features, not booleans")
    if _contains_text_values(values):
        raise ValueError("marks must contain real-valued numeric mark features, not text values")

    try:
        arr = np.asarray(values)
    except (TypeError, ValueError):
        return values
    if arr.dtype.kind == "c":
        if not _complex_has_zero_imaginary(arr):
            raise ValueError("marks must contain real-valued numeric mark features, not complex values")
        return np.real(arr)
    if arr.dtype == object:
        complex_entries = [
            item
            for item in arr.reshape(-1)
            if isinstance(item, (complex, np.complexfloating))
        ]
        if complex_entries:
            complex_values = np.asarray(complex_entries, dtype=complex)
            if not _complex_has_zero_imaginary(complex_values):
                raise ValueError("marks must contain real-valued numeric mark features, not complex values")
            normalized = arr.copy()
            for index in np.ndindex(normalized.shape):
                item = normalized[index]
                if isinstance(item, (complex, np.complexfloating)):
                    normalized[index] = float(np.real(item))
            return normalized
    return values


def _orient_mark_matrix_from_time_column(
    values: np.ndarray,
    *,
    spike_count: int,
    spike_times: np.ndarray,
    looks_like_time_column: Callable[[np.ndarray, np.ndarray], bool],
) -> np.ndarray:
    """Use a unique embedded time column to resolve an ambiguous spike axis."""

    if values.ndim < 2 or int(spike_count) <= 0:
        return values
    matching_axes = [axis for axis, size in enumerate(values.shape) if size == int(spike_count)]
    if len(matching_axes) < 2:
        return values

    time_aligned: list[np.ndarray] = []
    for axis in matching_axes:
        aligned = np.moveaxis(values, axis, 0).reshape(int(spike_count), -1)
        if aligned.shape[1] >= 2 and looks_like_time_column(aligned[:, 0], spike_times):
            time_aligned.append(aligned)
    return time_aligned[0] if len(time_aligned) == 1 else values


def _is_data_validation_wrapper(func: object) -> bool:
    return bool(getattr(func, _DATA_WRAPPER_ATTR, False))


def _is_clusterless_validation_wrapper(func: object) -> bool:
    return bool(getattr(func, _CLUSTERLESS_WRAPPER_ATTR, False))


def apply_mark_complex_validation_patch() -> None:
    """Install value-type validation for loaded and direct observation marks."""

    from . import clusterless, data

    current_coerce_mark_matrix = data._coerce_mark_matrix
    if not _is_data_validation_wrapper(current_coerce_mark_matrix):
        original_coerce_mark_matrix = current_coerce_mark_matrix

        @wraps(original_coerce_mark_matrix)
        def coerce_mark_matrix(value, *, spike_count: int, spike_times: np.ndarray):
            if _contains_boolean_values(value):
                return None
            arr = np.asarray(value)
            if arr.dtype.kind == "c":
                if not _complex_has_zero_imaginary(arr):
                    return None
                arr = np.real(arr)
            arr = _orient_mark_matrix_from_time_column(
                arr,
                spike_count=spike_count,
                spike_times=spike_times,
                looks_like_time_column=data._looks_like_time_column,
            )
            return original_coerce_mark_matrix(arr, spike_count=spike_count, spike_times=spike_times)

        setattr(coerce_mark_matrix, _DATA_WRAPPER_ATTR, True)
        data._coerce_mark_matrix = coerce_mark_matrix

    current_coerce_marks = clusterless.ClusterlessMarkEncoding._coerce_marks
    if not _is_clusterless_validation_wrapper(current_coerce_marks):
        original_coerce_marks = current_coerce_marks

        @wraps(original_coerce_marks)
        def coerce_marks(self, marks):
            return original_coerce_marks(self, _real_observation_marks_or_raise(marks))

        setattr(coerce_marks, _CLUSTERLESS_WRAPPER_ATTR, True)
        setattr(coerce_marks, "__hipporeplayimm_original__", original_coerce_marks)
        clusterless.ClusterlessMarkEncoding._coerce_marks = coerce_marks

    setattr(data, _PATCHED_FLAG, True)


__all__ = ["apply_mark_complex_validation_patch"]
