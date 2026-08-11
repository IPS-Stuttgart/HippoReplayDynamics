"""Prevent position interpolation outside measured contiguous tracking support."""

from __future__ import annotations

import sys
from functools import wraps

import numpy as np

_PATCH_MARKER = "_encoding_position_support_patch"
_PATCH_VERSION = 2
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_MAX_CONTIGUOUS_SAMPLE_GAP_MULTIPLIER = 5.0


def _synchronize_interpolator_aliases(previous: object, patched: object) -> None:
    """Refresh package-local aliases imported before the patch was installed."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_interp_positions", None) is previous:
            module._interp_positions = patched


def _max_contiguous_sample_gap_s(times: np.ndarray) -> float:
    """Return the largest gap still treated as continuously tracked."""

    values = np.asarray(times, dtype=float).reshape(-1)
    if values.size < 2:
        return float("inf")
    differences = np.diff(values)
    valid = differences[np.isfinite(differences) & (differences > 0.0)]
    if valid.size == 0:
        return float("inf")
    ordered = np.sort(valid)
    nominal_interval_s = float(ordered[(ordered.size - 1) // 2])
    return max(
        _MAX_CONTIGUOUS_SAMPLE_GAP_MULTIPLIER * nominal_interval_s,
        np.finfo(float).eps,
    )


def _queries_inside_tracking_gaps(
    times: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Return queries lying strictly inside oversized position-sample gaps."""

    time_values = np.asarray(times, dtype=float).reshape(-1)
    query_values = np.asarray(query_times, dtype=float).reshape(-1)
    inside_gap = np.zeros(query_values.shape, dtype=bool)
    if time_values.size < 2 or query_values.size == 0:
        return inside_gap

    differences = np.diff(time_values)
    max_sample_gap_s = _max_contiguous_sample_gap_s(time_values)
    for left_index in np.flatnonzero(differences > max_sample_gap_s):
        inside_gap |= (
            (query_values > time_values[left_index])
            & (query_values < time_values[left_index + 1])
        )
    return inside_gap


def apply_encoding_position_support_patch() -> None:
    """Mark position queries outside contiguous measured support as invalid."""

    from . import encoding

    current = encoding._interp_positions
    if getattr(current, _PATCH_MARKER, None) == _PATCH_VERSION:
        previous = getattr(current, _ORIGINAL_ATTR, None)
        if previous is not None:
            _synchronize_interpolator_aliases(previous, current)
        return

    previous = current

    @wraps(previous)
    def _interp_positions(
        times: np.ndarray,
        xy: np.ndarray,
        query_times: np.ndarray,
    ) -> np.ndarray:
        interpolated = np.asarray(previous(times, xy, query_times), dtype=float)
        time_values = np.asarray(times, dtype=float)
        query_values = np.asarray(query_times, dtype=float).reshape(-1)
        if time_values.ndim != 1 or time_values.shape[0] == 0:
            return interpolated
        if interpolated.ndim != 2 or interpolated.shape[0] != query_values.shape[0]:
            return interpolated

        outside_support = (
            ~np.isfinite(query_values)
            | (query_values < time_values[0])
            | (query_values > time_values[-1])
        )
        unsupported = outside_support | _queries_inside_tracking_gaps(
            time_values,
            query_values,
        )
        interpolated[unsupported] = np.nan
        return interpolated

    setattr(_interp_positions, _PATCH_MARKER, _PATCH_VERSION)
    setattr(_interp_positions, _ORIGINAL_ATTR, previous)
    encoding._interp_positions = _interp_positions
    _synchronize_interpolator_aliases(previous, _interp_positions)


__all__ = ["apply_encoding_position_support_patch"]
