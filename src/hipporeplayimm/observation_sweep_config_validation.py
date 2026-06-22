"""Runtime validation for observation-sweep parameter grids."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_observation_sweep_finite_config_validation_patch_applied"
_POSITIVE_GRID_FIELDS = (
    "bin_sizes_cm",
    "smoothing_sigmas_bins",
    "min_speed_cm_s",
    "min_occupancy_s",
    "rate_floor_hz",
    "time_bin_ms",
    "spike_rate_scales",
    "likelihood_temperatures",
)


def apply_observation_sweep_config_validation_patch() -> None:
    """Reject non-finite observation-sweep parameters before grid expansion."""

    from . import observation_sweep as sweep

    if getattr(sweep, _PATCHED_FLAG, False):
        return

    original_validate_config = sweep._validate_config

    @wraps(original_validate_config)
    def validate_config_with_finite_grid_values(config: Any) -> None:
        original_validate_config(config)
        _validate_finite_observation_sweep_config(config)

    sweep._validate_config = validate_config_with_finite_grid_values
    setattr(sweep, _PATCHED_FLAG, True)


def _validate_finite_observation_sweep_config(config: Any) -> None:
    for name in _POSITIVE_GRID_FIELDS:
        for value in getattr(config, name):
            numeric = _finite_float(name, value)
            if numeric <= 0.0:
                raise ValueError(f"{name} values must be positive")

    for value in getattr(config, "negative_binomial_overdispersions"):
        numeric = _finite_float("negative_binomial_overdispersions", value)
        if numeric < 0.0:
            raise ValueError("negative_binomial_overdispersions values must be nonnegative")

    decode_bin_s = _finite_float("decode_bin_s", getattr(config, "decode_bin_s"))
    if decode_bin_s <= 0.0:
        raise ValueError("decode_bin_s must be positive")


def _finite_float(name: str, value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be finite") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} values must be finite")
    return float(numeric)


__all__ = ["apply_observation_sweep_config_validation_patch"]
