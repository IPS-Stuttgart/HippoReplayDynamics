"""Compute place-field kinematics independently inside each behavioral run bout.

The sorted-spike and KD reference encoders both derive movement speed and frame
occupancy durations from the complete position time series before applying the
``run_times`` mask.  Large gaps between separate run bouts therefore leak into
``numpy.gradient`` and frame-duration estimates.  This runtime patch preserves
the existing encoder implementations while evaluating their private kinematic
helpers independently for every run interval.
"""

from __future__ import annotations

from functools import wraps
import sys
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
            speed[in_interval] = base_speed(times[in_interval], xy[in_interval])
    return speed


def _durations_within_run_intervals(
    times: np.ndarray,
    intervals: np.ndarray,
    base_durations: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    durations = np.zeros(times.shape, dtype=float)
    for start, end in intervals:
        in_interval = (times >= start) & (times <= end)
        if np.any(in_interval):
            durations[in_interval] = base_durations(times[in_interval])
    return durations


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
        raise RuntimeError("place-field encoder no longer exposes its kinematic helpers")

    intervals = _run_intervals(session.run_times)
    patched_globals = dict(function_globals)
    patched_globals["_speed_cm_s"] = lambda times, xy: _speed_within_run_intervals(
        times,
        xy,
        intervals,
        base_speed,
    )
    patched_globals["_frame_durations"] = lambda times: _durations_within_run_intervals(
        times,
        intervals,
        base_durations,
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


def _patch_encoder(module: Any, function_name: str) -> None:
    if getattr(module, _PATCHED_FLAG, False):
        return

    original = getattr(module, function_name)

    @wraps(original)
    def run_local_fit(session: Any, *args: Any, **kwargs: Any) -> Any:
        return _call_with_run_local_kinematics(original, session, *args, **kwargs)

    setattr(run_local_fit, _WRAPPER_MARKER, True)
    setattr(run_local_fit, _ORIGINAL_ATTR, original)
    setattr(module, function_name, run_local_fit)
    _synchronize_aliases(function_name, original, run_local_fit)
    setattr(module, _PATCHED_FLAG, True)


def apply_place_field_run_local_kinematics_patch() -> None:
    """Install run-local speed and occupancy durations on both encoders."""

    from . import encoding, kd_reference

    _patch_encoder(encoding, "fit_place_field_encoding")
    _patch_encoder(kd_reference, "fit_kd_place_field_encoding")


__all__ = ["apply_place_field_run_local_kinematics_patch"]
