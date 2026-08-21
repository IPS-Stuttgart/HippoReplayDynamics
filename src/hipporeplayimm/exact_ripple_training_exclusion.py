"""Apply exact half-open ripple exclusion to replay-training encoders."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import sys
from typing import Any, Callable

import numpy as np


_FRAME_DURATION_MARKER = "_exact_ripple_frame_duration_patch"
_EXCLUSION_INTERVAL_MARKER = "_exact_ripple_interval_source_patch"
_INTERVAL_MEMBERSHIP_MARKER = "_exact_ripple_interval_membership_patch"
_FIT_MARKER = "_exact_ripple_training_exclusion_patch"
_ORIGINAL_ATTR = "__hipporeplayimm_exact_ripple_original__"


class _TaggedExcludedIntervals(np.ndarray):
    """Array marker distinguishing ripple exclusions from behavioral intervals."""


@dataclass
class _TrainingExclusionContext:
    intervals: np.ndarray
    position_membership_handled: bool = False


_ACTIVE_EXCLUSION: ContextVar[_TrainingExclusionContext | None] = ContextVar(
    "hipporeplayimm_exact_training_exclusion",
    default=None,
)


def merge_half_open_intervals(intervals: Any) -> np.ndarray:
    """Validate and merge a union of half-open ``[start, end)`` intervals."""

    if intervals is None:
        return np.empty((0, 2), dtype=float)
    values = np.asarray(intervals, dtype=float)
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    if values.ndim == 1:
        if values.shape[0] != 2:
            raise ValueError("intervals must contain start/end pairs")
        values = values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("intervals must contain start/end pairs")
    if not np.all(np.isfinite(values)):
        raise ValueError("interval bounds must be finite")
    if np.any(values[:, 1] < values[:, 0]):
        raise ValueError("interval ends must not precede starts")

    nonempty = values[values[:, 1] > values[:, 0]]
    if nonempty.size == 0:
        return np.empty((0, 2), dtype=float)

    ordered = nonempty[np.lexsort((nonempty[:, 1], nonempty[:, 0]))]
    merged: list[list[float]] = [
        [float(ordered[0, 0]), float(ordered[0, 1])]
    ]
    for start, end in ordered[1:]:
        current = merged[-1]
        if float(start) <= current[1]:
            current[1] = max(current[1], float(end))
        else:
            merged.append([float(start), float(end)])
    return np.asarray(merged, dtype=float)


def retained_frame_durations(
    times: np.ndarray,
    durations: np.ndarray,
    excluded_intervals: Any,
) -> np.ndarray:
    """Subtract exact excluded overlap from timestamp-anchored frame exposure."""

    starts = np.asarray(times, dtype=float)
    retained = np.asarray(durations, dtype=float).copy()
    if starts.ndim != 1 or retained.shape != starts.shape:
        raise ValueError("times and durations must be matching one-dimensional arrays")
    if not np.all(np.isfinite(starts)):
        raise ValueError("frame times must be finite")
    if not np.all(np.isfinite(retained)) or np.any(retained < 0.0):
        raise ValueError("frame durations must be finite and nonnegative")

    intervals = merge_half_open_intervals(excluded_intervals)
    if intervals.size == 0 or retained.size == 0:
        return retained

    ends = starts + retained
    if not np.all(np.isfinite(ends)):
        raise ValueError("frame end times must be finite")
    for start, end in intervals:
        overlap = np.maximum(
            0.0,
            np.minimum(ends, end) - np.maximum(starts, start),
        )
        retained = np.maximum(retained - overlap, 0.0)
    return retained


def times_in_half_open_intervals(
    times: np.ndarray,
    intervals: Any,
) -> np.ndarray:
    """Return membership in a union of half-open ``[start, end)`` intervals."""

    values = np.asarray(times, dtype=float)
    merged = merge_half_open_intervals(intervals)
    mask = np.zeros(values.shape, dtype=bool)
    for start, end in merged:
        mask |= (values >= start) & (values < end)
    return mask


def apply_exact_ripple_training_exclusion_patch() -> None:
    """Install exact frame-overlap and half-open event exclusion on both encoders."""

    from . import clusterless, encoding

    _patch_shared_helpers(encoding)
    _patch_encoder(
        encoding,
        "fit_place_field_encoding",
        _standard_encoding_config,
    )
    _patch_encoder(
        clusterless,
        "fit_clusterless_mark_encoding",
        _clusterless_encoding_config,
    )


def _patch_shared_helpers(encoding: Any) -> None:
    _patch_frame_durations(encoding)
    _patch_exclusion_intervals(encoding)
    _patch_interval_membership(encoding)


def _patch_frame_durations(encoding: Any) -> None:
    current = encoding._frame_durations
    installed = _find_marked_wrapper(current, _FRAME_DURATION_MARKER)
    if installed is not None:
        _synchronize_wrapper_chain("_frame_durations", current, installed)
        return

    previous = current

    @wraps(previous, updated=())
    def frame_durations(times: np.ndarray) -> np.ndarray:
        durations = previous(times)
        context = _ACTIVE_EXCLUSION.get()
        if context is None:
            return durations
        return retained_frame_durations(
            times,
            durations,
            context.intervals,
        )

    setattr(frame_durations, _FRAME_DURATION_MARKER, True)
    setattr(frame_durations, _ORIGINAL_ATTR, previous)
    encoding._frame_durations = frame_durations
    _synchronize_aliases("_frame_durations", previous, frame_durations)


def _patch_exclusion_intervals(encoding: Any) -> None:
    current = encoding._encoding_exclusion_intervals
    installed = _find_marked_wrapper(current, _EXCLUSION_INTERVAL_MARKER)
    if installed is not None:
        _synchronize_wrapper_chain(
            "_encoding_exclusion_intervals",
            current,
            installed,
        )
        return

    previous = current

    @wraps(previous, updated=())
    def encoding_exclusion_intervals(session: Any, config: Any) -> np.ndarray:
        intervals = previous(session, config)
        if _ACTIVE_EXCLUSION.get() is None or np.asarray(intervals).size == 0:
            return intervals
        return np.asarray(intervals, dtype=float).view(_TaggedExcludedIntervals)

    setattr(encoding_exclusion_intervals, _EXCLUSION_INTERVAL_MARKER, True)
    setattr(encoding_exclusion_intervals, _ORIGINAL_ATTR, previous)
    encoding._encoding_exclusion_intervals = encoding_exclusion_intervals
    _synchronize_aliases(
        "_encoding_exclusion_intervals",
        previous,
        encoding_exclusion_intervals,
    )


def _patch_interval_membership(encoding: Any) -> None:
    current = encoding._times_in_intervals
    installed = _find_marked_wrapper(current, _INTERVAL_MEMBERSHIP_MARKER)
    if installed is not None:
        _synchronize_wrapper_chain("_times_in_intervals", current, installed)
        return

    previous = current

    @wraps(previous, updated=())
    def times_in_intervals(
        times: np.ndarray,
        intervals: np.ndarray,
    ) -> np.ndarray:
        context = _ACTIVE_EXCLUSION.get()
        if context is None or not isinstance(intervals, _TaggedExcludedIntervals):
            return previous(times, intervals)
        if not context.position_membership_handled:
            context.position_membership_handled = True
            return np.zeros(np.asarray(times).shape, dtype=bool)
        return times_in_half_open_intervals(times, context.intervals)

    setattr(times_in_intervals, _INTERVAL_MEMBERSHIP_MARKER, True)
    setattr(times_in_intervals, _ORIGINAL_ATTR, previous)
    encoding._times_in_intervals = times_in_intervals
    _synchronize_aliases("_times_in_intervals", previous, times_in_intervals)


def _patch_encoder(
    module: Any,
    function_name: str,
    config_getter: Callable[[Any], Any],
) -> None:
    current = getattr(module, function_name)
    installed = _find_marked_wrapper(current, _FIT_MARKER)
    if installed is not None:
        _synchronize_wrapper_chain(function_name, current, installed)
        return

    previous = current

    @wraps(previous, updated=())
    def fit_with_exact_exclusion(
        session: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        config = args[0] if args else kwargs.get("config")
        encoding_config = config_getter(config)
        enabled = _boolean_scalar(
            getattr(encoding_config, "exclude_ripple_intervals", False)
        )
        if enabled is not True:
            return previous(session, *args, **kwargs)

        try:
            from . import encoding

            intervals = encoding._ripple_intervals(session)
        except (AttributeError, TypeError, ValueError):
            return previous(session, *args, **kwargs)
        if intervals.size == 0:
            return previous(session, *args, **kwargs)

        token = _ACTIVE_EXCLUSION.set(
            _TrainingExclusionContext(np.asarray(intervals, dtype=float).copy())
        )
        try:
            return previous(session, *args, **kwargs)
        finally:
            _ACTIVE_EXCLUSION.reset(token)

    setattr(fit_with_exact_exclusion, _FIT_MARKER, True)
    setattr(fit_with_exact_exclusion, _ORIGINAL_ATTR, previous)
    setattr(module, function_name, fit_with_exact_exclusion)
    _synchronize_aliases(function_name, previous, fit_with_exact_exclusion)


def _standard_encoding_config(config: Any) -> Any:
    if config is not None:
        return config
    from .encoding import EncodingConfig

    return EncodingConfig()


def _clusterless_encoding_config(config: Any) -> Any:
    nested = None if config is None else getattr(config, "encoding", None)
    if nested is not None:
        return nested
    from .encoding import EncodingConfig

    return EncodingConfig()


def _boolean_scalar(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if (
        isinstance(value, np.ndarray)
        and value.ndim == 0
        and np.issubdtype(value.dtype, np.bool_)
    ):
        return bool(value.item())
    return None


def _find_marked_wrapper(function: Any, marker: str) -> Any | None:
    current = function
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        if getattr(current, marker, False):
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


def _synchronize_wrapper_chain(
    name: str,
    current: Any,
    installed: Any,
) -> None:
    original = getattr(installed, _ORIGINAL_ATTR, None)
    if callable(original):
        _synchronize_aliases(name, original, current)
    if installed is not current:
        _synchronize_aliases(name, installed, current)


def _synchronize_aliases(name: str, original: Any, replacement: Any) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name != "hipporeplayimm" and not module_name.startswith("hipporeplayimm."):
            continue
        if getattr(module, name, None) is original:
            setattr(module, name, replacement)


__all__ = [
    "apply_exact_ripple_training_exclusion_patch",
    "merge_half_open_intervals",
    "retained_frame_durations",
    "times_in_half_open_intervals",
]
