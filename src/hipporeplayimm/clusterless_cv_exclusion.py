"""Exact held-out interval exclusion for clusterless position validation."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np

_FRAME_HELPER_MARKER = "_clusterless_cv_frame_exclusion_helper"
_MARK_HELPER_MARKER = "_clusterless_cv_mark_exclusion_helper"


@dataclass(frozen=True)
class _ExclusionContext:
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
    removes spike marks in ``[start, end)`` while preserving the original run
    bouts for run-local speed estimation.
    """

    from . import clusterless

    intervals = _merge_half_open_intervals(excluded_intervals)
    if intervals.size == 0:
        return clusterless.fit_clusterless_mark_encoding(session, config)

    _install_contextual_helpers(clusterless)
    token = _EXCLUSION_CONTEXT.set(_ExclusionContext(excluded_intervals=intervals))
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

    current_training_marks = clusterless._training_marks
    if not getattr(current_training_marks, _MARK_HELPER_MARKER, False):

        @wraps(current_training_marks)
        def training_marks_without_held_out_intervals(*args: Any, **kwargs: Any):
            mark_times, mark_values, mark_group_ids = current_training_marks(
                *args,
                **kwargs,
            )
            context = _EXCLUSION_CONTEXT.get()
            if context is None:
                return mark_times, mark_values, mark_group_ids

            keep = ~_times_in_half_open_intervals(
                mark_times,
                context.excluded_intervals,
            )
            filtered_group_ids = (
                None
                if mark_group_ids is None
                else np.asarray(mark_group_ids)[keep]
            )
            return (
                np.asarray(mark_times)[keep],
                np.asarray(mark_values)[keep],
                filtered_group_ids,
            )

        setattr(training_marks_without_held_out_intervals, _MARK_HELPER_MARKER, True)
        clusterless._training_marks = training_marks_without_held_out_intervals


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


__all__ = ["fit_clusterless_mark_encoding_excluding_intervals"]
