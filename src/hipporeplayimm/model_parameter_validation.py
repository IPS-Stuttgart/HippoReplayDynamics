"""Reject boolean-valued numeric replay-model parameters."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_model_parameter_validation_patch_applied"


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


def apply_model_parameter_validation_patch() -> None:
    """Install strict bool rejection for replay-model numeric validators."""

    from . import models

    if getattr(models, _PATCHED_FLAG, False):
        return

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


__all__ = ["apply_model_parameter_validation_patch"]
