"""Strict numeric parsing guards for PyRecEst metadata and parameters."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_pyrecest_numeric_metadata_guard_applied"
_PARAMETER_PATCHED_FLAG = "_pyrecest_numeric_parameter_guard_applied"
_RAW_FLOAT_ERROR = "could not convert string to float"
_TEXT_SCALAR_TYPES = (str, bytes, np.str_, np.bytes_)


def _is_text_scalar(value: object) -> bool:
    if isinstance(value, _TEXT_SCALAR_TYPES):
        return True
    try:
        array = np.asarray(value)
    except ValueError:
        array = np.asarray(value, dtype=object)
    if array.shape != ():
        return False
    if array.dtype.kind in {"S", "U"}:
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), _TEXT_SCALAR_TYPES)
        except ValueError:
            return False
    return False


def _apply_pyrecest_metadata_guard_patch() -> None:
    from . import pyrecest_score_metadata as metadata

    current = metadata._metadata_float_from_value
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
    def metadata_float_from_value(value: Any, column: str) -> float | None:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{column} must contain finite numeric metadata values")
        try:
            return current(value, column)
        except ValueError as exc:
            if _RAW_FLOAT_ERROR in str(exc):
                raise ValueError(f"{column} must contain finite numeric metadata values") from exc
            raise

    setattr(metadata_float_from_value, _PATCHED_FLAG, True)
    setattr(metadata_float_from_value, "__hipporeplayimm_original__", current)
    metadata._metadata_float_from_value = metadata_float_from_value


def _apply_pyrecest_parameter_guard_patch() -> None:
    from . import pyrecest_models

    current = pyrecest_models._coerce_scalar_float
    if getattr(current, _PARAMETER_PATCHED_FLAG, False):
        return

    @wraps(current)
    def coerce_scalar_float(value: object, name: str, message: str) -> float:
        if _is_text_scalar(value):
            raise ValueError(message)
        return current(value, name, message)

    setattr(coerce_scalar_float, _PARAMETER_PATCHED_FLAG, True)
    setattr(coerce_scalar_float, "__hipporeplayimm_original__", current)
    pyrecest_models._coerce_scalar_float = coerce_scalar_float


def apply_pyrecest_numeric_metadata_guard_patch() -> None:
    """Reject malformed PyRecEst numeric metadata and text-valued parameters."""

    _apply_pyrecest_metadata_guard_patch()
    _apply_pyrecest_parameter_guard_patch()


__all__ = ["apply_pyrecest_numeric_metadata_guard_patch"]
