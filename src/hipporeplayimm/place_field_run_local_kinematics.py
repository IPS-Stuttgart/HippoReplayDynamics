"""Compute encoder kinematics independently inside each behavioral run bout.

The sorted-spike, KD reference, clusterless, and behavioral position-validation
paths derive movement speed and frame occupancy durations from position samples.
Large gaps between separate run bouts must not leak into ``numpy.gradient``,
occupancy durations, or training intervals.
"""

from __future__ import annotations

import sys
from functools import wraps
from types import FunctionType
from typing import Any, Callable

import numpy as np


_PATCHED_FLAG = "_place_field_run_local_kinematics_patch_applied"
_WRAPPER_MARKER = "_place_field_run_local_kinematics_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_run_local_kinematics_original__"


def _run_intervals(run_times: Any) -> np.ndarray:
    intervals = np.asarray(run_times, dtype=float)
    if intervals.size == 0:
        return np.empty((0, 2), dtype=float)
    intervals = np.atleast_2d(intervals)
    if intervals.shape[1] < 2:
        raise ValueError("run_times must contain start/end interval pairs")
    return intervals[:, :2]


def _speed_within_run_intervals(
    times: np.ndarray,
    xy: np.ndarray,
    intervals: np.ndarray,
    base_speed: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    xy = np.asarray(xy, dtype=float)
    speed = np.zeros(times.shape, dtype=float)
    for start, end in intervals:
        in_interval = (times >= start) & (times <= end)
        if np.any(in_interval):
            local_speed = base_speed(times[in_interval], xy[in_interval])
            # A frame can belong to overlapping or endpoint-sharing bouts.
            # Accumulate movement eligibility instead of letting interval order
            # decide which local derivative survives.
            speed[in_interval] = np.maximum(speed[in_interval], local_speed)
    return speed


def _interval_local_durations(
    times: np.ndarray,
    intervals: np.ndarray,
    base_durations: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Combine interval-local frame durations without row-order dependence.

    A sample shared by multiple run intervals can be terminal in one interval
    but have a real successor in another. Prefer the successor-derived duration
    over terminal median fallbacks; among equivalent candidates, keep the
    shortest duration so overlapping metadata cannot inflate occupancy.
    """

    exact = np.full(times.shape, np.inf, dtype=float)
    fallback = np.full(times.shape, np.inf, dtype=float)
    for start, end in intervals:
        indices = np.flatnonzero((times >= start) & (times <= end))
        if indices.size == 0:
            continue
        local = np.asarray(base_durations(times[indices]), dtype=float)
        if local.shape != indices.shape:
            raise ValueError("base_durations must return one value per input time")
        if indices.size > 1:
            np.minimum.at(exact, indices[:-1], local[:-1])
        np.minimum.at(fallback, indices[-1:], local[-1:])

    durations = np.zeros(times.shape, dtype=float)
    has_exact = np.isfinite(exact)
    durations[has_exact] = exact[has_exact]
    has_fallback = ~has_exact & np.isfinite(fallback)
    durations[has_fallback] = fallback[has_fallback]
    return durations


def _durations_within_run_intervals(
    times: np.ndarray,
    intervals: np.ndarray,
    base_durations: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    return _interval_local_durations(times, intervals, base_durations)


def _durations_split_at_run_boundaries(
    times: np.ndarray,
    intervals: np.ndarray,
    base_durations: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Compute frame durations per run bout while retaining out-of-run samples."""

    times = np.asarray(times, dtype=float)
    if times.size == 0 or intervals.size == 0:
        return base_durations(times)

    durations = _interval_local_durations(times, intervals, base_durations)
    covered = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        covered |= (times >= start) & (times <= end)

    outside = ~covered
    padded = np.concatenate(([False], outside, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(changes[0::2], changes[1::2], strict=True):
        durations[start:stop] = base_durations(times[start:stop])
    return durations


def _intervals_from_mask_and_durations(
    times: np.ndarray,
    mask: np.ndarray,
    durations: np.ndarray,
) -> np.ndarray:
    """Convert a frame mask to intervals using already-local frame durations."""

    times = np.asarray(times, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    durations = np.asarray(durations, dtype=float)
    if times.shape != mask.shape or times.shape != durations.shape:
        raise ValueError("times, mask, and durations must have matching shapes")
    if mask.size == 0 or not np.any(mask):
        return np.empty((0, 2), dtype=float)

    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    intervals = [
        [float(times[start]), float(times[stop - 1] + durations[stop - 1])]
        for start, stop in zip(changes[0::2], changes[1::2], strict=True)
    ]
    return np.asarray(intervals, dtype=float)


def _mask_intervals_within_run_intervals(
    times: np.ndarray,
    mask: np.ndarray,
    intervals: np.ndarray,
    base_durations: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Split masked training frames at behavioral run boundaries."""

    times = np.asarray(times, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if times.shape != mask.shape:
        raise ValueError("times and mask must have matching shapes")
    if mask.size == 0 or not np.any(mask):
        return np.empty((0, 2), dtype=float)
    if intervals.size == 0:
        return _intervals_from_mask_and_durations(
            times,
            mask,
            base_durations(times),
        )

    rows: list[np.ndarray] = []
    covered = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        in_interval = (times >= start) & (times <= end)
        covered |= in_interval
        if not np.any(mask & in_interval):
            continue
        local_times = times[in_interval]
        local_mask = mask[in_interval]
        local_rows = _intervals_from_mask_and_durations(
            local_times,
            local_mask,
            base_durations(local_times),
        )
        if local_rows.size:
            local_rows[:, 0] = np.maximum(local_rows[:, 0], float(start))
            local_rows[:, 1] = np.minimum(local_rows[:, 1], float(end))
            rows.append(local_rows)

    outside = ~covered
    padded = np.concatenate(([False], outside, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(changes[0::2], changes[1::2], strict=True):
        local_mask = mask[start:stop]
        if not np.any(local_mask):
            continue
        local_times = times[start:stop]
        rows.append(
            _intervals_from_mask_and_durations(
                local_times,
                local_mask,
                base_durations(local_times),
            )
        )

    nonempty = [row for row in rows if row.size]
    if not nonempty:
        return np.empty((0, 2), dtype=float)
    out = np.vstack(nonempty)
    return out[np.argsort(out[:, 0], kind="stable")]


def _call_with_run_local_kinematics(
    original: Callable[..., Any],
    session: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    function_globals = original.__globals__
    base_speed = function_globals.get("_speed_cm_s")
    base_durations = function_globals.get("_frame_durations")
    if not callable(base_speed) or not callable(base_durations):
        raise RuntimeError("kinematics caller no longer exposes its helper functions")

    intervals = _run_intervals(session.run_times)
    patched_globals = dict(function_globals)
    patched_globals["_speed_cm_s"] = lambda times, xy: _speed_within_run_intervals(
        times,
        xy,
        intervals,
        base_speed,
    )
    has_mask_intervals = callable(function_globals.get("_mask_to_intervals"))
    if has_mask_intervals:
        patched_globals["_frame_durations"] = lambda times: (
            _durations_split_at_run_boundaries(
                times,
                intervals,
                base_durations,
            )
        )
    else:
        patched_globals["_frame_durations"] = lambda times: (
            _durations_within_run_intervals(
                times,
                intervals,
                base_durations,
            )
        )
    if has_mask_intervals:
        patched_globals["_mask_to_intervals"] = lambda times, mask: (
            _mask_intervals_within_run_intervals(
                times,
                mask,
                intervals,
                base_durations,
            )
        )

    patched = FunctionType(
        original.__code__,
        patched_globals,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    patched.__kwdefaults__ = original.__kwdefaults__
    return patched(session, *args, **kwargs)


def _synchronize_aliases(function_name: str, original: Any, replacement: Any) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, function_name, None) is original:
            setattr(module, function_name, replacement)


def _find_run_local_wrapper(function: Any) -> Any | None:
    """Return the installed run-local wrapper from a ``__wrapped__`` chain."""

    current = function
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        if getattr(current, _WRAPPER_MARKER, False):
            return current
        seen.add(id(current))
        wrapped = getattr(current, "__wrapped__", None)
        if callable(wrapped):
            current = wrapped
            continue
        original = getattr(current, _ORIGINAL_ATTR, None)
        if callable(original):
            current = original
            continue
        break
    return None


def _patch_encoder(module: Any, function_name: str) -> None:
    current = getattr(module, function_name)
    installed = _find_run_local_wrapper(current)
    if installed is not None:
        original = getattr(installed, _ORIGINAL_ATTR, None)
        if original is not None:
            _synchronize_aliases(function_name, original, current)
        if installed is not current:
            _synchronize_aliases(function_name, installed, current)
        setattr(module, _PATCHED_FLAG, True)
        return

    original = current

    @wraps(original)
    def run_local_fit(session: Any, *args: Any, **kwargs: Any) -> Any:
        return _call_with_run_local_kinematics(original, session, *args, **kwargs)

    setattr(run_local_fit, _WRAPPER_MARKER, True)
    setattr(run_local_fit, _ORIGINAL_ATTR, original)
    setattr(module, function_name, run_local_fit)
    _synchronize_aliases(function_name, original, run_local_fit)
    setattr(module, _PATCHED_FLAG, True)


def apply_clusterless_run_local_kinematics_patch() -> None:
    """Install run-local speed and occupancy durations on the clusterless encoder."""

    from . import clusterless

    _patch_encoder(clusterless, "fit_clusterless_mark_encoding")


def apply_place_field_run_local_kinematics_patch() -> None:
    """Install run-local kinematics on encoders and position validation."""

    from . import encoding, kd_reference, position_validation

    _patch_encoder(encoding, "fit_place_field_encoding")
    _patch_encoder(kd_reference, "fit_kd_place_field_encoding")
    _patch_encoder(
        position_validation,
        "fit_place_field_encoding_for_position_mask",
    )
    _patch_encoder(position_validation, "validate_session_position_decoding")


__all__ = [
    "apply_clusterless_run_local_kinematics_patch",
    "apply_place_field_run_local_kinematics_patch",
]
