"""Runtime validation for observation-sweep parameter grids."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_observation_sweep_finite_config_validation_patch_applied"
_VALIDATOR_PATCHED_FLAG = "_observation_sweep_finite_config_validation_wrapper"
_SELECTION_PATCHED_FLAG = "_observation_calibration_gate_selection_patch_applied"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)
_POSITIVE_GRID_FIELDS = (
    "bin_sizes_cm",
    "min_occupancy_s",
    "rate_floor_hz",
    "time_bin_ms",
    "spike_rate_scales",
    "likelihood_temperatures",
)
_NONNEGATIVE_GRID_FIELDS = (
    "smoothing_sigmas_bins",
    "min_speed_cm_s",
    "negative_binomial_overdispersions",
)


def apply_observation_sweep_config_validation_patch() -> None:
    """Reject invalid sweep parameters and failed calibration selections."""

    from . import observation_sweep as sweep
    from . import result_quality_audit as audit

    current_validate_config = sweep._validate_config
    if not getattr(current_validate_config, _VALIDATOR_PATCHED_FLAG, False):

        @wraps(current_validate_config)
        def validate_config_with_finite_grid_values(config: Any) -> None:
            _validate_finite_observation_sweep_config(config)
            current_validate_config(config)

        setattr(validate_config_with_finite_grid_values, _VALIDATOR_PATCHED_FLAG, True)
        sweep._validate_config = validate_config_with_finite_grid_values
    setattr(sweep, _PATCHED_FLAG, True)

    _patch_observation_calibration_selection(audit)


def _patch_observation_calibration_selection(audit: Any) -> None:
    """Do not return rows that failed every configured calibration gate."""

    current_select = audit.select_observation_calibration
    if getattr(current_select, _SELECTION_PATCHED_FLAG, False):
        return

    @wraps(current_select)
    def select_observation_calibration(summary, config=None):
        selected = current_select(summary, config)
        if selected.empty or "selection_gate_passed" not in selected.columns:
            return selected
        if bool(audit._bool_series(selected["selection_gate_passed"]).any()):
            return selected
        return selected.iloc[:0].copy()

    setattr(select_observation_calibration, _SELECTION_PATCHED_FLAG, True)
    audit.select_observation_calibration = select_observation_calibration


def _validate_finite_observation_sweep_config(config: Any) -> None:
    _validate_sessions(getattr(config, "sessions", None))

    for name in _POSITIVE_GRID_FIELDS:
        for value in _grid_values(config, name):
            numeric = _finite_float(name, value)
            if numeric <= 0.0:
                raise ValueError(f"{name} values must be positive")

    for name in _NONNEGATIVE_GRID_FIELDS:
        for value in _grid_values(config, name):
            numeric = _finite_float(name, value)
            if numeric < 0.0:
                raise ValueError(f"{name} values must be nonnegative")

    decode_bin_s = _finite_float("decode_bin_s", getattr(config, "decode_bin_s"))
    if decode_bin_s <= 0.0:
        raise ValueError("decode_bin_s must be positive")

    _positive_integer("n_folds", getattr(config, "n_folds"))
    _positive_integer("simulation_events_per_model", getattr(config, "simulation_events_per_model"))


def _validate_sessions(sessions: Any) -> None:
    if sessions is None:
        return
    if isinstance(sessions, (str, bytes)):
        raise ValueError("sessions must be None or a non-empty sequence of session IDs")
    try:
        values = tuple(sessions)
    except TypeError as exc:
        raise ValueError("sessions must be None or a non-empty sequence of session IDs") from exc
    if not values:
        raise ValueError("sessions must be None or a non-empty sequence of session IDs")
    for session in values:
        if not isinstance(session, str) or not session.strip():
            raise ValueError("sessions must contain non-empty string session IDs")


def _grid_values(config: Any, name: str) -> tuple[Any, ...]:
    raw = getattr(config, name)
    if isinstance(raw, _STRING_TYPES):
        raise ValueError(f"{name} must be a non-empty sequence of numeric values")
    try:
        values = tuple(raw)
    except TypeError as exc:
        raise ValueError(f"{name} must contain at least one value") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} values must be finite scalars")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} values must be finite scalars") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} values must be finite scalars")
    item = array.item()
    if isinstance(item, _STRING_TYPES):
        raise ValueError(f"{name} values must be finite numeric scalars, not text")
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} values must be finite scalars") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} values must be finite scalars")
    return float(numeric)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} must be a positive integer")
    item = array.item()
    if isinstance(item, _STRING_TYPES):
        raise ValueError(f"{name} must be a positive integer, not text")
    try:
        integer = int(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    try:
        exactly_integral = bool(item == integer)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not exactly_integral or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


__all__ = ["apply_observation_sweep_config_validation_patch"]
