"""Validate replay-model numeric parameters."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_model_parameter_validation_patch_applied"
_REPLAY_CALIBRATION_PATCHED_FLAG = "_replay_calibration_max_gain_validation_patch_applied"


def _is_boolean_scalar(value: object) -> bool:
    """Return True for Python, NumPy, and object-wrapped boolean scalars."""

    if isinstance(value, (bool, np.bool_)):
        return True
    arr = np.asarray(value)
    if arr.ndim != 0:
        return False
    if np.issubdtype(arr.dtype, np.bool_):
        return True
    if arr.dtype == object:
        try:
            return isinstance(arr.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _reject_boolean_scalar(name: str, value: object) -> None:
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not boolean")


def _validate_replay_calibration_max_gain(calibration: object | None) -> None:
    if calibration is None:
        return
    try:
        max_gain = float(getattr(calibration, "max_gain"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("max_gain must be finite and at least 1.0") from exc
    if not np.isfinite(max_gain) or max_gain < 1.0:
        raise ValueError("max_gain must be finite and at least 1.0")


def _apply_replay_calibration_max_gain_validation_patch() -> None:
    from . import result_improvement_extensions as extensions

    current = extensions.build_sorted_emissions_with_replay_calibration
    if getattr(current, _REPLAY_CALIBRATION_PATCHED_FLAG, False):
        return

    @wraps(current)
    def build_sorted_emissions_with_replay_calibration(session, encoding, ripple, config=None, calibration=None):
        _validate_replay_calibration_max_gain(calibration)
        return current(session, encoding, ripple, config, calibration)

    setattr(build_sorted_emissions_with_replay_calibration, _REPLAY_CALIBRATION_PATCHED_FLAG, True)
    setattr(build_sorted_emissions_with_replay_calibration, "__hipporeplayimm_original__", current)
    extensions.build_sorted_emissions_with_replay_calibration = build_sorted_emissions_with_replay_calibration


def apply_model_parameter_validation_patch() -> None:
    """Install strict numeric validation patches for replay-model parameters."""

    from . import models

    if not getattr(models, _PATCHED_FLAG, False):
        original_positive = models._validate_positive_parameter
        original_nonnegative = models._validate_nonnegative_parameter
        original_probability = models._validate_probability_parameter

        @wraps(original_positive)
        def validate_positive_parameter(name: str, value: float) -> None:
            _reject_boolean_scalar(name, value)
            return original_positive(name, value)

        @wraps(original_nonnegative)
        def validate_nonnegative_parameter(name: str, value: float) -> None:
            _reject_boolean_scalar(name, value)
            return original_nonnegative(name, value)

        @wraps(original_probability)
        def validate_probability_parameter(name: str, value: float) -> None:
            _reject_boolean_scalar(name, value)
            return original_probability(name, value)

        models._validate_positive_parameter = validate_positive_parameter
        models._validate_nonnegative_parameter = validate_nonnegative_parameter
        models._validate_probability_parameter = validate_probability_parameter
        setattr(models, _PATCHED_FLAG, True)

    _apply_replay_calibration_max_gain_validation_patch()


__all__ = ["apply_model_parameter_validation_patch"]
