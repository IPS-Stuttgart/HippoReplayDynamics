"""Prevent position interpolation and validation windows outside contiguous tracking support."""

from __future__ import annotations

import sys
from functools import wraps

import numpy as np

_PATCH_MARKER = "_encoding_position_support_patch"
_PATCH_VERSION = 2
_DECODE_WINDOWS_PATCH_MARKER = "_position_decoding_tracking_support_patch"
_DECODE_WINDOWS_PATCH_VERSION = 1
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


def _synchronize_decode_window_aliases(previous: object, patched: object) -> None:
    """Refresh package-local decode-window aliases imported before patching."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_decode_windows", None) is previous:
            module._decode_windows = patched


def _max_contiguous_sample_gap_s(times: np.ndarray) -> float:
    """Return the largest gap still treated as continuously tracked."""

    values = np.asarray(times, dtype=float).reshape(-1)
    if values.size < 2:
        return float("inf")
    differences = np.diff(values)
    valid = differences[np.isfinite(differences) & (differences > 0.0)]
    if valid.size == 0:
        return float("inf")
    nominal_interval_s = float(np.median(valid))
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


def _tracking_support_intervals(
    times: np.ndarray,
    run_times: np.ndarray,
) -> np.ndarray:
    """Return run intervals clipped and split to contiguous measured support."""

    time_values = np.asarray(times, dtype=float).reshape(-1)
    if time_values.size == 0:
        return np.empty((0, 2), dtype=float)

    intervals = np.asarray(run_times, dtype=float)
    if intervals.size == 0:
        intervals = np.array([[time_values[0], time_values[-1]]], dtype=float)
    else:
        intervals = np.atleast_2d(intervals)
        if intervals.shape[1] < 2:
            raise ValueError("run_times must contain start/end interval pairs")
        intervals = intervals[:, :2]

    differences = np.diff(time_values)
    valid_differences = differences[
        np.isfinite(differences) & (differences > 0.0)
    ]
    nominal_interval_s = (
        float(np.median(valid_differences))
        if valid_differences.size
        else 0.0
    )
    max_sample_gap_s = _max_contiguous_sample_gap_s(time_values)
    rows: list[list[float]] = []

    for start, end in intervals:
        if end <= start:
            continue
        indices = np.flatnonzero((time_values >= start) & (time_values <= end))
        if indices.size == 0:
            continue
        local_differences = np.diff(time_values[indices])
        breaks = np.flatnonzero(
            ~np.isfinite(local_differences)
            | (local_differences <= 0.0)
            | (local_differences > max_sample_gap_s)
        ) + 1
        for segment in np.split(indices, breaks):
            if segment.size == 0:
                continue
            segment_times = time_values[segment]
            segment_differences = np.diff(segment_times)
            finite_positive = segment_differences[
                np.isfinite(segment_differences) & (segment_differences > 0.0)
            ]
            terminal_interval_s = (
                float(np.median(finite_positive))
                if finite_positive.size
                else nominal_interval_s
            )
            support_start = max(float(start), float(segment_times[0]))
            support_end = min(
                float(end),
                float(segment_times[-1] + terminal_interval_s),
            )
            if support_end > support_start:
                rows.append([support_start, support_end])

    return np.asarray(rows, dtype=float).reshape(-1, 2)


def _patch_position_decode_windows_if_loaded() -> None:
    """Restrict behavioral validation windows to measured tracking support."""

    module = sys.modules.get("hipporeplayimm.position_validation")
    if module is None:
        return

    current = getattr(module, "_decode_windows", None)
    if not callable(current):
        return
    if getattr(current, _DECODE_WINDOWS_PATCH_MARKER, None) == _DECODE_WINDOWS_PATCH_VERSION:
        previous = getattr(current, _ORIGINAL_ATTR, None)
        if previous is not None:
            _synchronize_decode_window_aliases(previous, current)
        return

    previous = current

    @wraps(previous)
    def _decode_windows(
        times: np.ndarray,
        xy: np.ndarray,
        movement: np.ndarray,
        run_times: np.ndarray,
        decode_bin_s: float,
    ) -> list[dict[str, float]]:
        supported_run_times = _tracking_support_intervals(times, run_times)
        if supported_run_times.size == 0:
            return []
        return previous(
            times,
            xy,
            movement,
            supported_run_times,
            decode_bin_s,
        )

    setattr(
        _decode_windows,
        _DECODE_WINDOWS_PATCH_MARKER,
        _DECODE_WINDOWS_PATCH_VERSION,
    )
    setattr(_decode_windows, _ORIGINAL_ATTR, previous)
    module._decode_windows = _decode_windows
    _synchronize_decode_window_aliases(previous, _decode_windows)


def apply_encoding_position_support_patch() -> None:
    """Mark position queries and validation windows outside support as invalid."""

    from . import encoding

    current = encoding._interp_positions
    if getattr(current, _PATCH_MARKER, None) == _PATCH_VERSION:
        previous = getattr(current, _ORIGINAL_ATTR, None)
        if previous is not None:
            _synchronize_interpolator_aliases(previous, current)
        _patch_position_decode_windows_if_loaded()
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
    _patch_position_decode_windows_if_loaded()


__all__ = ["apply_encoding_position_support_patch"]
