"""Runtime validation for observation-sweep parameter grids."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_observation_sweep_finite_config_validation_patch_applied"
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
    """Reject invalid observation-sweep parameters before grid expansion."""

    from . import observation_sweep as sweep

    if getattr(sweep, _PATCHED_FLAG, False):
        return

    original_validate_config = sweep._validate_config

    @wraps(original_validate_config)
    def validate_config_with_finite_grid_values(config: Any) -> None:
        _validate_finite_observation_sweep_config(config)
        original_validate_config(config)

    sweep._validate_config = validate_config_with_finite_grid_values
    setattr(sweep, _PATCHED_FLAG, True)


def _validate_finite_observation_sweep_config(config: Any) -> None:
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


def _grid_values(config: Any, name: str) -> tuple[Any, ...]:
    try:
        values = tuple(getattr(config, name))
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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be finite scalars") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} values must be finite scalars")
    try:
        numeric = float(array)
    except (TypeError, ValueError) as exc:
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
    try:
        numeric = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


__all__ = ["apply_observation_sweep_config_validation_patch"]
