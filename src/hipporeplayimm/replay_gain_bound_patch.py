"""Keep replay-calibration cell gains within their configured bounds.

The result-improvement calibration separates a global event gain from relative
per-cell gains by forcing the cell-gain geometric mean to one.  Clipping before
that centering is insufficient: dividing by the geometric mean can move a gain
outside ``[1 / max_gain, max_gain]``.  This patch performs the centering in log
space while projecting onto the bounded zero-mean interval.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_replay_gain_bound_patch_applied"
_WRAPPER_FLAG = "_bounded_replay_gain_wrapper"
_WRAPPER_VERSION = 3
_ORIGINAL_ATTR = "__hipporeplayimm_replay_gain_bound_original__"


def _validated_max_gain(max_gain: float) -> float:
    """Return a finite gain cap whose reciprocal interval contains one."""

    bound = float(max_gain)
    if not np.isfinite(bound) or bound < 1.0:
        raise ValueError("max_gain must be finite and greater than or equal to 1")
    return bound


def _bounded_geometric_center(gains: Any, max_gain: float) -> np.ndarray:
    """Return bounded positive gains with geometric mean exactly one.

    The projection has the form ``clip(log(gain) - offset, -L, L)`` with
    ``L = log(max_gain)``.  Its mean is monotone in ``offset``, so bisection
    finds the unique centered solution while retaining the original ordering.
    """

    values = np.asarray(gains, dtype=float)
    bound = _validated_max_gain(max_gain)
    if values.size == 0:
        return values.copy()

    log_bound = float(np.log(bound))
    if log_bound == 0.0:
        return np.ones_like(values, dtype=float)

    clipped = np.clip(values, 1.0 / bound, bound)
    log_values = np.log(clipped)

    lower = float(np.min(log_values) - log_bound)
    upper = float(np.max(log_values) + log_bound)
    for _ in range(80):
        offset = 0.5 * (lower + upper)
        centered = np.clip(log_values - offset, -log_bound, log_bound)
        if float(np.mean(centered)) > 0.0:
            lower = offset
        else:
            upper = offset

    centered = np.clip(
        log_values - 0.5 * (lower + upper),
        -log_bound,
        log_bound,
    )
    return np.exp(centered)


def _apply_replay_gains_with_bounds(
    rates_hz: np.ndarray,
    counts: np.ndarray,
    bin_durations: np.ndarray,
    *,
    mode: str,
    prior_count: float,
    max_gain: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Apply event and cell gains without violating the cell-gain cap."""

    bound = _validated_max_gain(max_gain)
    if mode == "none":
        return rates_hz, {
            "replay_event_gain": 1.0,
            "replay_cell_gain_geomean": 1.0,
            "replay_cell_gain_min": 1.0,
            "replay_cell_gain_max": 1.0,
        }

    calibrated = np.asarray(rates_hz, dtype=float).copy()
    durations = np.asarray(bin_durations, dtype=float)
    expected_by_cell = np.sum(
        calibrated * durations[None, :],
        axis=1,
        dtype=float,
    )
    expected_by_cell = np.maximum(expected_by_cell, np.finfo(float).tiny)
    observed_by_cell = np.asarray(counts).sum(axis=0).astype(float)

    cell_gains = np.ones(calibrated.shape[0], dtype=float)
    if mode in {"cell", "event-cell"} and calibrated.shape[0]:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            raw_cell_gains = (observed_by_cell + prior_count) / (
                expected_by_cell + prior_count
            )
        cell_gains = _bounded_geometric_center(raw_cell_gains, bound)
        calibrated *= cell_gains[:, None]

    event_gain = 1.0
    if mode in {"event", "event-cell"}:
        expected_total = max(
            float(
                np.sum(
                    calibrated * durations[None, :],
                    dtype=float,
                )
            ),
            np.finfo(float).tiny,
        )
        observed_total = float(observed_by_cell.sum())
        event_gain = (observed_total + prior_count) / (
            expected_total + prior_count
        )
        event_gain = float(
            np.clip(event_gain, 1.0 / bound, bound)
        )
        calibrated *= event_gain

    return np.maximum(calibrated, np.finfo(float).tiny), {
        "replay_event_gain": float(event_gain),
        "replay_cell_gain_geomean": (
            float(np.exp(np.mean(np.log(cell_gains))))
            if cell_gains.size
            else 1.0
        ),
        "replay_cell_gain_min": (
            float(np.min(cell_gains)) if cell_gains.size else 1.0
        ),
        "replay_cell_gain_max": (
            float(np.max(cell_gains)) if cell_gains.size else 1.0
        ),
    }


def apply_replay_gain_bound_patch() -> None:
    """Install bounded geometric centering for result-improvement gains."""

    from . import result_improvement_extensions

    current = result_improvement_extensions._apply_replay_gains
    if getattr(current, _WRAPPER_FLAG, None) == _WRAPPER_VERSION:
        setattr(result_improvement_extensions, _PATCHED_FLAG, True)
        return

    @wraps(current)
    def apply_replay_gains(
        rates_hz,
        counts,
        bin_durations,
        *,
        mode,
        prior_count,
        max_gain,
    ):
        return _apply_replay_gains_with_bounds(
            rates_hz,
            counts,
            bin_durations,
            mode=mode,
            prior_count=prior_count,
            max_gain=max_gain,
        )

    setattr(apply_replay_gains, _WRAPPER_FLAG, _WRAPPER_VERSION)
    setattr(apply_replay_gains, _ORIGINAL_ATTR, current)
    result_improvement_extensions._apply_replay_gains = apply_replay_gains
    setattr(result_improvement_extensions, _PATCHED_FLAG, True)


__all__ = ["apply_replay_gain_bound_patch"]
