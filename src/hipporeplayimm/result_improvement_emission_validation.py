"""Validate replay-calibrated result-improvement emission parameters."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_result_improvement_emission_validation_patch_applied"
_BUILD_SORTED_EMISSIONS_WRAPPER_FLAG = "_replay_calibrated_emission_parameter_validation_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_emission_validation_original__"


def _is_boolean_scalar(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _is_boolean_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim == 0:
        return False
    if np.issubdtype(array.dtype, np.bool_):
        return True
    if array.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in array.flat)
    return False


def _is_text_scalar(value: object) -> bool:
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), (str, bytes, np.str_, np.bytes_))
        except ValueError:
            return False
    return False


def _is_text_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim == 0:
        return False
    if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
        return True
    if array.dtype == object:
        return any(isinstance(item, (str, bytes, np.str_, np.bytes_)) for item in array.flat)
    return False


def _reject_array_shaped_scalar(name: str, value: object) -> None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric scalar") from exc
    if array.ndim != 0:
        raise TypeError(f"{name} must be a numeric scalar")
    if _is_text_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not text")


def _reject_boolean_scalar(name: str, value: object) -> None:
    _reject_array_shaped_scalar(name, value)
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not boolean")


def _reject_boolean_numeric(name: str, value: object) -> None:
    if _is_boolean_scalar(value) or _is_boolean_array(value):
        raise TypeError(f"{name} must be numeric, not boolean")
    if _is_text_scalar(value) or _is_text_array(value):
        raise TypeError(f"{name} must be numeric, not text")


def _finite_positive_scalar(name: str, value: object) -> float:
    _reject_boolean_scalar(name, value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _finite_nonnegative_scalar(name: str, value: object) -> float:
    _reject_boolean_scalar(name, value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _max_gain_scalar(value: object) -> float:
    _reject_boolean_scalar("max_gain", value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_gain must be finite and at least 1.0") from exc
    if not np.isfinite(numeric) or numeric < 1.0:
        raise ValueError("max_gain must be finite and at least 1.0")
    return numeric


def _required_attr(source: object, name: str) -> Any:
    try:
        return getattr(source, name)
    except AttributeError as exc:
        raise ValueError(f"{name} must be provided") from exc


def _validate_replay_calibrated_emission_parameters(config: object | None, calibration: object | None) -> None:
    """Reject malformed scalars before the builder's historical ``float(...)`` coercions."""

    if config is not None:
        _finite_positive_scalar("time_bin_s", _required_attr(config, "time_bin_s"))
        _finite_positive_scalar("spike_rate_scale", _required_attr(config, "spike_rate_scale"))
        _finite_positive_scalar("likelihood_temperature", _required_attr(config, "likelihood_temperature"))
        _finite_nonnegative_scalar(
            "negative_binomial_overdispersion",
            _required_attr(config, "negative_binomial_overdispersion"),
        )
        cell_weights = getattr(config, "cell_weights", None)
        if cell_weights is not None:
            _reject_boolean_numeric("cell_weights", cell_weights)

    if calibration is not None:
        _finite_nonnegative_scalar("gain_prior_count", _required_attr(calibration, "gain_prior_count"))
        _max_gain_scalar(_required_attr(calibration, "max_gain"))
        _finite_positive_scalar(
            "negative_binomial_dispersion",
            _required_attr(calibration, "negative_binomial_dispersion"),
        )


def apply_result_improvement_emission_validation_patch() -> None:
    """Install strict scalar guards for replay-calibrated sorted-spike emissions."""

    from . import result_improvement_extensions

    current = result_improvement_extensions.build_sorted_emissions_with_replay_calibration
    if getattr(current, _BUILD_SORTED_EMISSIONS_WRAPPER_FLAG, False):
        setattr(result_improvement_extensions, _PATCHED_FLAG, True)
        return

    @wraps(current)
    def build_sorted_emissions_with_replay_calibration(session, encoding, ripple, config=None, calibration=None):
        _validate_replay_calibrated_emission_parameters(config, calibration)
        return current(session, encoding, ripple, config, calibration)

    setattr(build_sorted_emissions_with_replay_calibration, _BUILD_SORTED_EMISSIONS_WRAPPER_FLAG, True)
    setattr(build_sorted_emissions_with_replay_calibration, _ORIGINAL_ATTR, current)
    result_improvement_extensions.build_sorted_emissions_with_replay_calibration = build_sorted_emissions_with_replay_calibration
    setattr(result_improvement_extensions, _PATCHED_FLAG, True)


__all__ = ["apply_result_improvement_emission_validation_patch"]
