"""Validate mark-matrix candidates before numeric coercion.

Spike marks are analog waveform or amplitude features.  Complex values with a
nonzero imaginary component, and boolean values that are likely logical metadata
rather than marks, must be rejected before the legacy coercion path can silently
convert them to real-valued features.  Manual ``SpikeMarkData`` objects may also
store a single mark feature as a one-dimensional vector; normalize that shape
before feature counting or clusterless mark-likelihood construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_mark_complex_validation_patch_applied"
_COERCE_MARK_MATRIX_WRAPPER_ATTR = "_mark_complex_validation_wrapper"
_N_SPIKES_WRAPPER_ATTR = "_spike_mark_n_spikes_shape_wrapper"
_N_FEATURES_WRAPPER_ATTR = "_spike_mark_n_features_shape_wrapper"
_ALL_EVENT_MARKS_WRAPPER_ATTR = "_clusterless_all_event_marks_shape_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


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


def _is_wrapper(func: object, attr: str) -> bool:
    return bool(getattr(func, attr, False))


def _mark_wrapper(wrapper: Any, original: Any, attr: str) -> Any:
    if callable(original):
        wrapper = wraps(original)(wrapper)
    setattr(wrapper, attr, True)
    setattr(wrapper, _ORIGINAL_ATTR, original)
    return wrapper


def _spike_mark_spike_count(marks: Any) -> int:
    arr = np.asarray(marks)
    if arr.size == 0:
        return 0
    if arr.ndim == 0:
        return 1
    return int(arr.shape[0])


def _spike_mark_feature_count(marks: Any) -> int:
    arr = np.asarray(marks)
    if arr.size == 0:
        return 0
    if arr.ndim <= 1:
        return 1
    return int(arr.shape[1])


def _mark_values_as_2d(value: Any) -> np.ndarray:
    marks = np.asarray(value, dtype=float)
    if marks.ndim == 0:
        return marks.reshape(1, 1)
    if marks.ndim == 1:
        return marks.reshape(-1, 1)
    return marks


def apply_mark_complex_validation_patch() -> None:
    """Install value-type validation and mark-shape normalization hooks."""

    from . import clusterless, data

    _patch_coerce_mark_matrix(data)
    _patch_spike_mark_shape_properties(data)
    _patch_clusterless_event_mark_shape(clusterless)
    setattr(data, _PATCHED_FLAG, True)
    setattr(clusterless, _PATCHED_FLAG, True)


def _patch_coerce_mark_matrix(data: Any) -> None:
    current_coerce_mark_matrix = data._coerce_mark_matrix
    if _is_wrapper(current_coerce_mark_matrix, _COERCE_MARK_MATRIX_WRAPPER_ATTR):
        return

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

    setattr(coerce_mark_matrix, _COERCE_MARK_MATRIX_WRAPPER_ATTR, True)
    setattr(coerce_mark_matrix, _ORIGINAL_ATTR, original_coerce_mark_matrix)
    data._coerce_mark_matrix = coerce_mark_matrix


def _patch_spike_mark_shape_properties(data: Any) -> None:
    current_n_spikes = getattr(getattr(data.SpikeMarkData, "n_spikes", None), "fget", None)
    current_n_features = getattr(getattr(data.SpikeMarkData, "n_features", None), "fget", None)

    if not _is_wrapper(current_n_spikes, _N_SPIKES_WRAPPER_ATTR):
        def n_spikes(self):
            return _spike_mark_spike_count(self.marks)

        data.SpikeMarkData.n_spikes = property(
            _mark_wrapper(n_spikes, current_n_spikes, _N_SPIKES_WRAPPER_ATTR)
        )

    if not _is_wrapper(current_n_features, _N_FEATURES_WRAPPER_ATTR):
        def n_features(self):
            return _spike_mark_feature_count(self.marks)

        data.SpikeMarkData.n_features = property(
            _mark_wrapper(n_features, current_n_features, _N_FEATURES_WRAPPER_ATTR)
        )


def _patch_clusterless_event_mark_shape(clusterless: Any) -> None:
    current_all_event_marks = clusterless._all_event_marks
    if _is_wrapper(current_all_event_marks, _ALL_EVENT_MARKS_WRAPPER_ATTR):
        return

    @wraps(current_all_event_marks)
    def all_event_marks(session):
        marks = session.spike_marks
        if marks is None:
            raise ValueError("Session does not contain spike marks.")
        mark_times = np.asarray(marks.times, dtype=float)
        mark_values = _mark_values_as_2d(marks.marks)
        if mark_values.shape[0] != mark_times.shape[0]:
            raise ValueError("spike marks must contain one row per spike-mark time")
        return mark_times, mark_values

    setattr(all_event_marks, _ALL_EVENT_MARKS_WRAPPER_ATTR, True)
    setattr(all_event_marks, _ORIGINAL_ATTR, current_all_event_marks)
    clusterless._all_event_marks = all_event_marks


__all__ = ["apply_mark_complex_validation_patch"]
