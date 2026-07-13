"""Validate mark-matrix candidates before numeric coercion.

Spike marks are analog waveform or amplitude features.  Complex values with a
nonzero imaginary component, and boolean values that are likely logical metadata
rather than marks, must be rejected before the legacy coercion path can silently
convert them to real-valued features.  Ambiguous square mark matrices are
reoriented when an embedded spike-time column uniquely identifies the spike axis.
Tetrode/cell mapping tables are likewise oriented only when the observed spike
cell IDs identify one functional mapping direction unambiguously.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCHED_FLAG = "_mark_complex_validation_patch_applied"
_PATCH_WRAPPER_ATTR = "_mark_complex_validation_wrapper"
_TETRODE_ORIENTATION_WRAPPER_ATTR = "_tetrode_cell_id_orientation_wrapper"


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


def _mapping_is_functional(pairs: np.ndarray, *, cell_column: int) -> bool:
    """Return whether one candidate cell column maps each ID to one group."""

    group_column = 1 - cell_column
    mapping: dict[int, int] = {}
    for row in pairs:
        cell_id = int(row[cell_column])
        group_id = int(row[group_column])
        previous = mapping.get(cell_id)
        if previous is not None and previous != group_id:
            return False
        mapping[cell_id] = group_id
    return True


def _resolve_tetrode_cell_id_orientation(
    cell_ids: Any,
    tetrode_cell_ids: Any,
    *,
    coerce_integral_ids: Callable[[Any, str], np.ndarray],
) -> Any:
    """Resolve equal-overlap mapping orientation without an arbitrary tie break."""

    if cell_ids is None:
        return tetrode_cell_ids
    arr = np.asarray(tetrode_cell_ids)
    if arr.size == 0:
        return tetrode_cell_ids
    normalized = np.squeeze(arr)
    if normalized.ndim != 2:
        return tetrode_cell_ids
    if normalized.shape[0] == 2 and normalized.shape[1] != 2:
        normalized = normalized.T
    if normalized.shape[1] < 2:
        return tetrode_cell_ids

    raw_pairs = np.asarray(normalized[:, :2])
    try:
        numeric_pairs = np.asarray(raw_pairs, dtype=float)
    except (TypeError, ValueError):
        return tetrode_cell_ids
    finite_rows = np.isfinite(numeric_pairs).all(axis=1)
    if not np.any(finite_rows):
        return tetrode_cell_ids

    pairs = coerce_integral_ids(raw_pairs[finite_rows], "tetrode/cell IDs").reshape(-1, 2)
    spike_cell_ids = coerce_integral_ids(cell_ids, "spike cell IDs").reshape(-1)
    unique_spike_cells = np.unique(spike_cell_ids)
    first_matches = int(np.isin(unique_spike_cells, pairs[:, 0]).sum())
    second_matches = int(np.isin(unique_spike_cells, pairs[:, 1]).sum())
    if first_matches == 0 or first_matches != second_matches:
        return tetrode_cell_ids

    first_is_functional = _mapping_is_functional(pairs, cell_column=0)
    second_is_functional = _mapping_is_functional(pairs, cell_column=1)
    if first_is_functional and not second_is_functional:
        oriented = np.array(normalized, copy=True)
        oriented[:, [0, 1]] = oriented[:, [1, 0]]
        return oriented
    if second_is_functional and not first_is_functional:
        return tetrode_cell_ids
    if first_is_functional and second_is_functional:
        raise ValueError(
            "ambiguous Tetrode_Cell_IDs orientation: both columns match spike cell IDs equally"
        )
    return tetrode_cell_ids


def _is_mark_complex_validation_wrapper(func: object) -> bool:
    return bool(getattr(func, _PATCH_WRAPPER_ATTR, False))


def _is_tetrode_orientation_wrapper(func: object) -> bool:
    return bool(getattr(func, _TETRODE_ORIENTATION_WRAPPER_ATTR, False))


def apply_mark_complex_validation_patch() -> None:
    """Install mark-matrix and tetrode/cell orientation validation."""

    from . import data, data_cell_id_validation

    current_coerce_mark_matrix = data._coerce_mark_matrix
    current_mark_group_ids = data._mark_group_ids_from_tetrode_cell_ids
    if (
        _is_mark_complex_validation_wrapper(current_coerce_mark_matrix)
        and _is_tetrode_orientation_wrapper(current_mark_group_ids)
    ):
        setattr(data, _PATCHED_FLAG, True)
        return

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
                arr = np.real(arr)
            arr = _orient_mark_matrix_from_time_column(
                arr,
                spike_count=spike_count,
                spike_times=spike_times,
                looks_like_time_column=data._looks_like_time_column,
            )
            return original_coerce_mark_matrix(arr, spike_count=spike_count, spike_times=spike_times)

        setattr(coerce_mark_matrix, _PATCH_WRAPPER_ATTR, True)
        data._coerce_mark_matrix = coerce_mark_matrix

    if not _is_tetrode_orientation_wrapper(current_mark_group_ids):
        original_mark_group_ids = current_mark_group_ids

        @wraps(original_mark_group_ids)
        def mark_group_ids_from_tetrode_cell_ids(cell_ids, tetrode_cell_ids):
            oriented = _resolve_tetrode_cell_id_orientation(
                cell_ids,
                tetrode_cell_ids,
                coerce_integral_ids=data_cell_id_validation._coerce_integral_ids,
            )
            return original_mark_group_ids(cell_ids, oriented)

        setattr(
            mark_group_ids_from_tetrode_cell_ids,
            _TETRODE_ORIENTATION_WRAPPER_ATTR,
            True,
        )
        data._mark_group_ids_from_tetrode_cell_ids = mark_group_ids_from_tetrode_cell_ids

    setattr(data, _PATCHED_FLAG, True)


__all__ = ["apply_mark_complex_validation_patch"]
