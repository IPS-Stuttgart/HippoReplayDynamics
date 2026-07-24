"""Encoding runtime patches for grid compatibility, kinematics, and validation."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from .place_field_run_local_kinematics import apply_place_field_run_local_kinematics_patch


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
_GRID_ORIGINAL_ATTR = "_encoding_grid_extra_columns_original_make_grid"
_GRID_WRAPPER_MARKER = "_encoding_grid_extra_columns_wrapper"
_SPEED_ORIGINAL_ATTR = "_encoding_nonuniform_time_original_speed"
_SPEED_WRAPPER_MARKER = "_encoding_nonuniform_time_speed_wrapper"
_BOOL_ORIGINALS_ATTR = "_encoding_bool_validation_originals"
_AS_XY_ARRAY_WRAPPER_MARKER = "_encoding_numeric_as_xy_array_wrapper"
_AS_POSITION_ARRAY_WRAPPER_MARKER = "_encoding_numeric_as_position_array_wrapper"
_VALIDATE_ENCODING_CONFIG_WRAPPER_MARKER = "_encoding_bool_validate_encoding_config_wrapper"
_TIME_BIN_EDGES_WRAPPER_MARKER = "_encoding_bool_time_bin_edges_wrapper"
_TIME_BIN_EDGES_WRAPPER_VERSION = 2
_POISSON_LOG_EMISSIONS_WRAPPER_MARKER = "_encoding_bool_poisson_log_emissions_wrapper"


def apply_encoding_grid_extra_columns_patch() -> None:
    """Install encoding compatibility patches."""

    from . import encoding

    _apply_grid_extra_columns_patch(encoding)
    _apply_nonuniform_time_speed_patch(encoding)
    _apply_encoding_bool_validation_patch(encoding)
    apply_place_field_run_local_kinematics_patch()


def _apply_grid_extra_columns_patch(encoding) -> None:
    original_make_grid = _original_make_grid(encoding)
    if getattr(encoding, "_grid_extra_columns_patch_applied", False) and _is_marked_wrapper(
        getattr(encoding, "_make_grid", None),
        _GRID_WRAPPER_MARKER,
    ):
        return

    def make_grid(xy, config):
        arr = encoding._as_xy_array(xy, name="xy")
        return original_make_grid(arr[:, :2], config)

    _copy_function_metadata(original_make_grid, make_grid)
    _mark_wrapper(make_grid, _GRID_WRAPPER_MARKER)
    encoding._make_grid = make_grid
    _synchronize_make_grid_aliases(original_make_grid, make_grid)
    encoding._grid_extra_columns_patch_applied = True


def _apply_nonuniform_time_speed_patch(encoding) -> None:
    """Differentiate positions against timestamp coordinates, not sample indices."""

    current_speed = encoding._speed_cm_s
    original_speed = getattr(encoding, _SPEED_ORIGINAL_ATTR, None)
    if original_speed is None:
        original_speed = current_speed
        setattr(encoding, _SPEED_ORIGINAL_ATTR, original_speed)

    if _is_marked_wrapper(current_speed, _SPEED_WRAPPER_MARKER):
        _synchronize_speed_aliases(original_speed, current_speed)
        return

    def speed_cm_s(times, xy):
        time_values = np.asarray(times, dtype=float)
        positions = np.asarray(xy, dtype=float)
        if time_values.shape[0] < 2:
            return np.zeros(time_values.shape, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            dx_dt = np.gradient(positions[:, 0], time_values)
            dy_dt = np.gradient(positions[:, 1], time_values)
            speed = np.hypot(dx_dt, dy_dt)
        return np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

    _copy_function_metadata(original_speed, speed_cm_s)
    _mark_wrapper(speed_cm_s, _SPEED_WRAPPER_MARKER)
    encoding._speed_cm_s = speed_cm_s
    _synchronize_speed_aliases(original_speed, speed_cm_s)


def _apply_encoding_bool_validation_patch(encoding) -> None:
    originals = _encoding_bool_originals(encoding)
    if getattr(encoding, "_encoding_bool_validation_patch_applied", False) and _encoding_bool_wrappers_are_current(encoding):
        _synchronize_encoding_validation_aliases(encoding, originals)
        return

    original_as_xy_array = originals["as_xy_array"]
    original_as_position_array = originals["as_position_array"]
    original_validate_encoding_config = originals["validate_encoding_config"]
    original_time_bin_edges = originals["time_bin_edges"]
    original_poisson_log_emissions = originals["poisson_log_emissions"]

    def as_xy_array(xy, *, name="xy"):
        _require_numeric_values(xy, name)
        return original_as_xy_array(xy, name=name)

    def as_position_array(position):
        _require_numeric_values(position, "position")
        return original_as_position_array(position)

    def validate_encoding_config(config):
        for name in _NUMERIC_ENCODING_CONFIG_FIELDS:
            _require_numeric_scalar(getattr(config, name), name)
        for name in _BOOLEAN_ENCODING_CONFIG_FIELDS:
            _require_boolean_scalar(getattr(config, name), name)
        return original_validate_encoding_config(config)

    def time_bin_edges(start, end, time_bin_s):
        _require_numeric_scalar(start, "ripple start")
        _require_numeric_scalar(end, "ripple end")
        _require_numeric_scalar(time_bin_s, "time_bin_s")
        _validate_time_bin_range(start, end, time_bin_s)
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
        _require_numeric_values(dt, "dt")
        _require_numeric_scalar(spike_rate_scale, "spike_rate_scale")
        _require_numeric_scalar(likelihood_temperature, "likelihood_temperature")
        _require_numeric_scalar(negative_binomial_overdispersion, "negative_binomial_overdispersion")
        if checked_cell_weights is not None:
            _require_numeric_values(checked_cell_weights, "cell_weights")
        return original_poisson_log_emissions(
            spike_counts,
            rates_hz,
            dt,
            spike_rate_scale=spike_rate_scale,
            likelihood_temperature=likelihood_temperature,
            cell_weights=checked_cell_weights,
            negative_binomial_overdispersion=negative_binomial_overdispersion,
        )

    _copy_function_metadata(original_as_xy_array, as_xy_array)
    _copy_function_metadata(original_as_position_array, as_position_array)
    _copy_function_metadata(original_validate_encoding_config, validate_encoding_config)
    _copy_function_metadata(original_time_bin_edges, time_bin_edges)
    _copy_function_metadata(original_poisson_log_emissions, poisson_log_emissions)
    _mark_wrapper(as_xy_array, _AS_XY_ARRAY_WRAPPER_MARKER)
    _mark_wrapper(as_position_array, _AS_POSITION_ARRAY_WRAPPER_MARKER)
    _mark_wrapper(validate_encoding_config, _VALIDATE_ENCODING_CONFIG_WRAPPER_MARKER)
    setattr(time_bin_edges, _TIME_BIN_EDGES_WRAPPER_MARKER, _TIME_BIN_EDGES_WRAPPER_VERSION)
    _mark_wrapper(poisson_log_emissions, _POISSON_LOG_EMISSIONS_WRAPPER_MARKER)
    encoding._as_xy_array = as_xy_array
    encoding._as_position_array = as_position_array
    encoding._validate_encoding_config = validate_encoding_config
    encoding._time_bin_edges = time_bin_edges
    encoding._poisson_log_emissions = poisson_log_emissions
    _synchronize_encoding_validation_aliases(encoding, originals)
    encoding._encoding_bool_validation_patch_applied = True


def _original_make_grid(encoding) -> Any:
    original = getattr(encoding, _GRID_ORIGINAL_ATTR, None)
    if original is None:
        original = encoding._make_grid
        setattr(encoding, _GRID_ORIGINAL_ATTR, original)
    return original


def _encoding_bool_originals(encoding) -> dict[str, Any]:
    originals = getattr(encoding, _BOOL_ORIGINALS_ATTR, None)
    if originals is None:
        originals = {
            "as_xy_array": encoding._as_xy_array,
            "as_position_array": encoding._as_position_array,
            "validate_encoding_config": encoding._validate_encoding_config,
            "time_bin_edges": encoding._time_bin_edges,
            "poisson_log_emissions": encoding._poisson_log_emissions,
        }
        setattr(encoding, _BOOL_ORIGINALS_ATTR, originals)
    return originals


def _encoding_bool_wrappers_are_current(encoding) -> bool:
    return (
        _is_marked_wrapper(
            getattr(encoding, "_as_xy_array", None),
            _AS_XY_ARRAY_WRAPPER_MARKER,
        )
        and _is_marked_wrapper(
            getattr(encoding, "_as_position_array", None),
            _AS_POSITION_ARRAY_WRAPPER_MARKER,
        )
        and _is_marked_wrapper(
            getattr(encoding, "_validate_encoding_config", None),
            _VALIDATE_ENCODING_CONFIG_WRAPPER_MARKER,
        )
        and getattr(
            getattr(encoding, "_time_bin_edges", None),
            _TIME_BIN_EDGES_WRAPPER_MARKER,
            None,
        )
        == _TIME_BIN_EDGES_WRAPPER_VERSION
        and _is_marked_wrapper(
            getattr(encoding, "_poisson_log_emissions", None),
            _POISSON_LOG_EMISSIONS_WRAPPER_MARKER,
        )
    )


def _mark_wrapper(wrapper: Any, marker: str) -> Any:
    setattr(wrapper, marker, True)
    return wrapper


def _is_marked_wrapper(value: Any, marker: str) -> bool:
    return bool(getattr(value, marker, False))


def _reject_boolean_numeric(value: Any, name: str) -> None:
    if _contains_boolean(value):
        raise TypeError(f"{name} must be numeric, not boolean")


def _require_numeric_scalar(value: Any, name: str) -> None:
    array = _as_real_numeric_array(value, name)
    if array.ndim != 0:
        raise TypeError(f"{name} must be a numeric scalar")


def _require_numeric_values(value: Any, name: str) -> None:
    _as_real_numeric_array(value, name)


def _validate_time_bin_range(start: Any, end: Any, time_bin_s: Any) -> None:
    """Reject otherwise-valid ripple spans that overflow bin arithmetic."""

    start_value = float(np.asarray(start).item())
    end_value = float(np.asarray(end).item())
    bin_width = float(np.asarray(time_bin_s).item())
    if (
        not np.isfinite(start_value)
        or not np.isfinite(end_value)
        or not np.isfinite(bin_width)
        or bin_width <= 0.0
        or end_value <= start_value
    ):
        return

    with np.errstate(over="ignore", invalid="ignore"):
        duration = end_value - start_value
    if not np.isfinite(duration):
        raise ValueError("ripple duration exceeds floating-point range")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        complete_bins = np.floor(duration / bin_width)
    if (
        not np.isfinite(complete_bins)
        or complete_bins > np.iinfo(np.intp).max - 1
    ):
        raise ValueError("ripple time-bin count exceeds platform index range")


def _as_real_numeric_array(value: Any, name: str) -> np.ndarray:
    _reject_boolean_numeric(value, name)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc

    if array.dtype == object:
        if not all(_is_real_numeric_value(item) for item in array.flat):
            raise TypeError(f"{name} must be numeric")
        return array

    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    return array


def _is_real_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


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


def _synchronize_speed_aliases(original_speed, replacement_speed) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_speed_cm_s", None) is original_speed:
            module._speed_cm_s = replacement_speed


def _synchronize_encoding_validation_aliases(encoding, originals: dict[str, Any]) -> None:
    replacements = {
        "_as_xy_array": encoding._as_xy_array,
        "_as_position_array": encoding._as_position_array,
        "_validate_encoding_config": encoding._validate_encoding_config,
        "_time_bin_edges": encoding._time_bin_edges,
        "_poisson_log_emissions": encoding._poisson_log_emissions,
    }
    original_by_attr = {
        "_as_xy_array": originals["as_xy_array"],
        "_as_position_array": originals["as_position_array"],
        "_validate_encoding_config": originals["validate_encoding_config"],
        "_time_bin_edges": originals["time_bin_edges"],
        "_poisson_log_emissions": originals["poisson_log_emissions"],
    }
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        for attr, original in original_by_attr.items():
            if getattr(module, attr, None) is original:
                setattr(module, attr, replacements[attr])
