"""Validate mark-matrix candidates with complex dtype."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_mark_complex_validation_patch_applied"
_PATCH_WRAPPER_ATTR = "_mark_complex_validation_wrapper"


def _complex_has_zero_imaginary(values: np.ndarray) -> bool:
    imaginary = np.imag(values)
    return bool(np.all(np.isfinite(imaginary)) and np.allclose(imaginary, 0.0, rtol=0.0, atol=0.0))


def _is_mark_complex_validation_wrapper(func: object) -> bool:
    return bool(getattr(func, _PATCH_WRAPPER_ATTR, False))


def apply_mark_complex_validation_patch() -> None:
    """Install complex-value validation for mark-matrix candidates."""

    from . import data

    current_coerce_mark_matrix = data._coerce_mark_matrix
    if _is_mark_complex_validation_wrapper(current_coerce_mark_matrix):
        setattr(data, _PATCHED_FLAG, True)
        return

    original_coerce_mark_matrix = current_coerce_mark_matrix

    @wraps(original_coerce_mark_matrix)
    def coerce_mark_matrix(value, *, spike_count: int, spike_times: np.ndarray):
        arr = np.asarray(value)
        if arr.dtype.kind == "c":
            if not _complex_has_zero_imaginary(arr):
                return None
            value = np.real(arr)
        return original_coerce_mark_matrix(value, spike_count=spike_count, spike_times=spike_times)

    setattr(coerce_mark_matrix, _PATCH_WRAPPER_ATTR, True)
    data._coerce_mark_matrix = coerce_mark_matrix
    setattr(data, _PATCHED_FLAG, True)


__all__ = ["apply_mark_complex_validation_patch"]
