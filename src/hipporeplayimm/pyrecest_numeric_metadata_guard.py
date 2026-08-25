"""Strict numeric parsing guards for PyRecEst metadata and parameters."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_FLAG = "_pyrecest_numeric_metadata_guard_applied"
_PARAMETER_PATCHED_FLAG = "_pyrecest_numeric_parameter_guard_applied"
_RAW_FLOAT_ERROR = "could not convert string to float"
_TEXT_SCALAR_TYPES = (str, bytes, bytearray, memoryview, np.str_, np.bytes_)


def _unwrap_zero_dimensional_scalar(value: object) -> tuple[object, bool]:
    """Expose the semantic leaf of nested 0-D object arrays.

    MATLAB/HDF5 and pandas-adjacent inputs can contain scalar values wrapped in
    more than one zero-dimensional object array. Python's ``float`` recursively
    unwraps such arrays, so validation has to inspect the same semantic leaf
    before numeric coercion is allowed.
    """

    current = value
    seen: set[int] = set()
    while isinstance(current, np.ndarray) and current.ndim == 0:
        marker = id(current)
        if marker in seen:
            return current, True
        seen.add(marker)
        try:
            nested = current.item()
        except (TypeError, ValueError):
            return current, False
        if nested is current:
            return current, True
        current = nested
    return current, False


def _as_scalar_array(value: object) -> np.ndarray:
    try:
        return np.asarray(value)
    except ValueError:
        return np.asarray(value, dtype=object)


def _is_text_scalar(value: object) -> bool:
    current, cyclic = _unwrap_zero_dimensional_scalar(value)
    if cyclic:
        return True
    if isinstance(current, _TEXT_SCALAR_TYPES):
        return True
    array = _as_scalar_array(current)
    return array.shape == () and array.dtype.kind in {"S", "U"}


def _is_boolean_scalar(value: object) -> bool:
    current, cyclic = _unwrap_zero_dimensional_scalar(value)
    if cyclic:
        return False
    if isinstance(current, (bool, np.bool_)):
        return True
    array = _as_scalar_array(current)
    return array.shape == () and np.issubdtype(array.dtype, np.bool_)


def _is_complex_scalar(value: object) -> bool:
    current, cyclic = _unwrap_zero_dimensional_scalar(value)
    if cyclic:
        return False
    if isinstance(current, (complex, np.complexfloating)):
        return True
    array = _as_scalar_array(current)
    return array.shape == () and np.issubdtype(array.dtype, np.complexfloating)


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

    # PyRecEst's validators resolve this helper through module globals. Replacing
    # it closes the same nested-object loophole for Boolean parameters before
    # the scalar float coercer is reached.
    pyrecest_models._is_boolean_scalar = _is_boolean_scalar

    current = pyrecest_models._coerce_scalar_float
    if getattr(current, _PARAMETER_PATCHED_FLAG, False):
        return

    @wraps(current)
    def coerce_scalar_float(value: object, name: str, message: str) -> float:
        if _is_text_scalar(value) or _is_complex_scalar(value):
            raise ValueError(message)
        return current(value, name, message)

    setattr(coerce_scalar_float, _PARAMETER_PATCHED_FLAG, True)
    setattr(coerce_scalar_float, "__hipporeplayimm_original__", current)
    pyrecest_models._coerce_scalar_float = coerce_scalar_float


def apply_pyrecest_numeric_metadata_guard_patch() -> None:
    """Reject malformed PyRecEst numeric metadata and semantic scalar aliases."""

    _apply_pyrecest_metadata_guard_patch()
    _apply_pyrecest_parameter_guard_patch()


__all__ = ["apply_pyrecest_numeric_metadata_guard_patch"]
