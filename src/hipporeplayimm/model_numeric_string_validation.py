"""Reject string-valued replay-model numeric parameters."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCHED_FLAG = "_model_numeric_string_validation_patch_applied"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)
_VALIDATOR_NAMES = (
    "_validate_positive_parameter",
    "_validate_nonnegative_parameter",
    "_validate_probability_parameter",
)


def _is_string_scalar(value: object) -> bool:
    if isinstance(value, _STRING_TYPES):
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
            return isinstance(array.item(), _STRING_TYPES)
        except ValueError:
            return False
    return False


def _reject_string_scalar(name: str, value: object) -> None:
    if _is_string_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not string")


def apply_model_numeric_string_validation_patch() -> None:
    """Install string-scalar guards around model parameter validators."""

    from . import models

    if getattr(models, _PATCHED_FLAG, False):
        return

    for validator_name in _VALIDATOR_NAMES:
        current = getattr(models, validator_name)

        @wraps(current)
        def validator(name: str, value: object, *, _current=current):
            _reject_string_scalar(name, value)
            return _current(name, value)

        setattr(validator, _PATCHED_FLAG, True)
        setattr(validator, "__hipporeplayimm_original__", current)
        setattr(models, validator_name, validator)

    setattr(models, _PATCHED_FLAG, True)


__all__ = ["apply_model_numeric_string_validation_patch"]
