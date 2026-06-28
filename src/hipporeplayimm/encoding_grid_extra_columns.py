"""Encoding runtime patches for grid compatibility and numeric validation."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np


_NUMERIC_ENCODING_CONFIG_FIELDS = (
    "bin_size_cm",
    "smoothing_sigma_bins",
    "min_speed_cm_s",
    "min_occupancy_s",
    "rate_floor_hz",
    "arena_padding_cm",
)
_BOOLEAN_ENCODING_CONFIG_FIELDS = (
    "use_excitatory",
    "exclude_ripple_intervals",
)


def apply_encoding_grid_extra_columns_patch() -> None:
    """Install encoding compatibility patches."""

    from . import encoding

    _apply_grid_extra_columns_patch(encoding)
    _apply_encoding_bool_validation_patch(encoding)


def _apply_grid_extra_columns_patch(encoding) -> None:
    if getattr(encoding, "_grid_extra_columns_patch_applied", False):
        return

    original_make_grid = encoding._make_grid

    def make_grid(xy, config):
        arr = encoding._as_xy_array(xy, name="xy")
        return original_make_grid(arr[:, :2], config)

    _copy_function_metadata(original_make_grid, make_grid)
    encoding._make_grid = make_grid
    _synchronize_make_grid_aliases(original_make_grid, make_grid)
    encoding._grid_extra_columns_patch_applied = True


def _apply_encoding_bool_validation_patch(encoding) -> None:
    if getattr(encoding, "_encoding_bool_validation_patch_applied", False):
        return

    original_validate_encoding_config = encoding._validate_encoding_config
    original_time_bin_edges = encoding._time_bin_edges
    original_poisson_log_emissions = encoding._poisson_log_emissions

    def validate_encoding_config(config):
        for name in _NUMERIC_ENCODING_CONFIG_FIELDS:
            _reject_boolean_numeric(getattr(config, name), name)
        for name in _BOOLEAN_ENCODING_CONFIG_FIELDS:
            _require_boolean_scalar(getattr(config, name), name)
        return original_validate_encoding_config(config)

    def time_bin_edges(start, end, time_bin_s):
        _reject_boolean_numeric(time_bin_s, "time_bin_s")
        return original_time_bin_edges(start, end, time_bin_s)

    def poisson_log_emissions(
        spike_counts,
        rates_hz,
        dt,
        *,
        spike_rate_scale=1.0,
        likelihood_temperature=1.0,
        cell_weights=None,
        negative_binomial_overdispersion=0.0,
    ):
        checked_cell_weights = _materialize_iterable_for_validation(cell_weights)
        _reject_boolean_numeric(dt, "dt")
        _reject_boolean_numeric(spike_rate_scale, "spike_rate_scale")
        _reject_boolean_numeric(likelihood_temperature, "likelihood_temperature")
        _reject_boolean_numeric(negative_binomial_overdispersion, "negative_binomial_overdispersion")
        if checked_cell_weights is not None:
            _reject_boolean_numeric(checked_cell_weights, "cell_weights")
        return original_poisson_log_emissions(
            spike_counts,
            rates_hz,
            dt,
            spike_rate_scale=spike_rate_scale,
            likelihood_temperature=likelihood_temperature,
            cell_weights=checked_cell_weights,
            negative_binomial_overdispersion=negative_binomial_overdispersion,
        )

    _copy_function_metadata(original_validate_encoding_config, validate_encoding_config)
    _copy_function_metadata(original_time_bin_edges, time_bin_edges)
    _copy_function_metadata(original_poisson_log_emissions, poisson_log_emissions)
    encoding._validate_encoding_config = validate_encoding_config
    encoding._time_bin_edges = time_bin_edges
    encoding._poisson_log_emissions = poisson_log_emissions
    encoding._encoding_bool_validation_patch_applied = True


def _reject_boolean_numeric(value: Any, name: str) -> None:
    if _contains_boolean(value):
        raise TypeError(f"{name} must be numeric, not boolean")


def _require_boolean_scalar(value: Any, name: str) -> None:
    if not _is_boolean_scalar(value):
        raise TypeError(f"{name} must be boolean")


def _is_boolean_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if not isinstance(value, np.ndarray):
        return False
    if value.ndim != 0:
        return False
    if np.issubdtype(value.dtype, np.bool_):
        return True
    if value.dtype == object:
        try:
            return isinstance(value.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean(item) for item in value.flat)
        return False
    if np.isscalar(value):
        return False
    try:
        iterator = iter(value)
    except TypeError:
        return False
    return any(_contains_boolean(item) for item in iterator)


def _materialize_iterable_for_validation(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, np.ndarray)) or np.isscalar(value):
        return value
    try:
        return list(value)
    except TypeError:
        return value


def _copy_function_metadata(original, replacement) -> None:
    replacement.__name__ = original.__name__
    replacement.__doc__ = original.__doc__
    replacement.__module__ = original.__module__


def _synchronize_make_grid_aliases(original_make_grid, replacement_make_grid) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_make_grid", None) is original_make_grid:
            module._make_grid = replacement_make_grid
