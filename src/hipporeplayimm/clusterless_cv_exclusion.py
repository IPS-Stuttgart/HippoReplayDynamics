"""Exact held-out interval exclusion for clusterless position validation."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np

from .encoding import _clean_position

_FRAME_HELPER_MARKER = "_clusterless_cv_frame_exclusion_helper"
_INTERVAL_HELPER_MARKER = "_clusterless_cv_interval_exclusion_helper"


@dataclass(frozen=True)
class _ExclusionContext:
    position_times: np.ndarray
    run_times: np.ndarray
    excluded_intervals: np.ndarray


_EXCLUSION_CONTEXT: ContextVar[_ExclusionContext | None] = ContextVar(
    "clusterless_cv_exclusion_context",
    default=None,
)


def fit_clusterless_mark_encoding_excluding_intervals(
    session: Any,
    config: Any,
    excluded_intervals: np.ndarray,
) -> Any:
    """Fit clusterless encoding without using held-out half-open intervals.

    Position samples represent frames whose exposure can straddle a validation
    boundary. The ordinary timestamp mask therefore cannot remove the exact
    held-out exposure. This helper subtracts overlap from frame durations and
    excludes spike marks in ``[start, end)`` while preserving the session's
    original run bouts for run-local speed estimation.
    """

    from . import clusterless

    intervals = _merge_half_open_intervals(excluded_intervals)
    if intervals.size == 0:
        return clusterless.fit_clusterless_mark_encoding(session, config)

    _install_contextual_helpers(clusterless)
    position = _clean_position(session.position)
    context = _ExclusionContext(
        position_times=np.asarray(position[:, 0], dtype=float),
        run_times=np.asarray(session.run_times, dtype=float),
        excluded_intervals=intervals,
    )
    token = _EXCLUSION_CONTEXT.set(context)
    try:
        return clusterless.fit_clusterless_mark_encoding(session, config)
    finally:
        _EXCLUSION_CONTEXT.reset(token)


def _install_contextual_helpers(clusterless: Any) -> None:
    current_durations = clusterless._frame_durations
    if not getattr(current_durations, _FRAME_HELPER_MARKER, False):

        @wraps(current_durations)
        def frame_durations_without_held_out_exposure(times: np.ndarray) -> np.ndarray:
            durations = np.asarray(current_durations(times), dtype=float)
            context = _EXCLUSION_CONTEXT.get()
            if context is None:
                return durations
            return _exclude_half_open_frame_durations(
                times,
                durations,
                context.excluded_intervals,
            )

        setattr(frame_durations_without_held_out_exposure, _FRAME_HELPER_MARKER, True)
        clusterless._frame_durations = frame_durations_without_held_out_exposure

    current_membership = clusterless._times_in_intervals
    if not getattr(current_membership, _INTERVAL_HELPER_MARKER, False):

        @wraps(current_membership)
        def interval_membership_without_held_out_marks(
            times: np.ndarray,
            intervals: np.ndarray,
        ) -> np.ndarray:
            membership = np.asarray(current_membership(times, intervals), dtype=bool)
            context = _EXCLUSION_CONTEXT.get()
            if context is None or not _same_numeric_array(intervals, context.run_times):
                return membership

            numeric_times = np.asarray(times, dtype=float)
            if _same_numeric_array(numeric_times, context.position_times):
                return membership
            return membership & ~_times_in_half_open_intervals(
                numeric_times,
                context.excluded_intervals,
            )

        setattr(interval_membership_without_held_out_marks, _INTERVAL_HELPER_MARKER, True)
        clusterless._times_in_intervals = interval_membership_without_held_out_marks


def _exclude_half_open_frame_durations(
    times: np.ndarray,
    durations: np.ndarray,
    excluded_intervals: np.ndarray,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    durations = np.asarray(durations, dtype=float)
    if times.shape != durations.shape:
        raise ValueError("times and durations must have matching shapes")

    retained = durations.copy()
    frame_ends = times + durations
    for start, end in np.asarray(excluded_intervals, dtype=float).reshape(-1, 2):
        overlap = np.maximum(
            np.minimum(frame_ends, end) - np.maximum(times, start),
            0.0,
        )
        retained -= overlap
    return np.maximum(retained, 0.0)


def _times_in_half_open_intervals(
    times: np.ndarray,
    intervals: np.ndarray,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    membership = np.zeros(times.shape, dtype=bool)
    for start, end in np.asarray(intervals, dtype=float).reshape(-1, 2):
        membership |= (times >= start) & (times < end)
    return membership


def _merge_half_open_intervals(intervals: np.ndarray) -> np.ndarray:
    values = np.asarray(intervals, dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    values = values.reshape(-1, 2)
    valid = values[
        np.isfinite(values).all(axis=1)
        & (values[:, 1] > values[:, 0])
    ]
    if valid.size == 0:
        return np.empty((0, 2), dtype=float)

    valid = valid[np.argsort(valid[:, 0], kind="stable")]
    merged: list[list[float]] = [[float(valid[0, 0]), float(valid[0, 1])]]
    for start, end in valid[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(end))
        else:
            merged.append([float(start), float(end)])
    return np.asarray(merged, dtype=float)


def _same_numeric_array(left: Any, right: Any) -> bool:
    try:
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False
    return left_array.shape == right_array.shape and np.array_equal(
        left_array,
        right_array,
        equal_nan=True,
    )


__all__ = ["fit_clusterless_mark_encoding_excluding_intervals"]
