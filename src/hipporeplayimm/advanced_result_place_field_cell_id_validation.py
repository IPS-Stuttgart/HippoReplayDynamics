"""Validate place-field diagnostic inputs before numeric coercion."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_place_field_cell_id_validation_patch_applied"
_BASE_ATTR = "_place_field_cell_id_validation_base"
_MISSING_CELL_ID_STRINGS = {"", "nan", "na", "n/a", "none", "null", "<na>"}


def _is_missing_scalar(value: object) -> bool:
    """Return True only when pandas reports a scalar missing value."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _contains_boolean(value: object) -> bool:
    """Return whether an array-like input contains a boolean scalar."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return value.size > 0
        if value.dtype == object:
            return any(_contains_boolean(item) for item in value.flat)
        return False
    if isinstance(value, (pd.Series, pd.Index)):
        return any(_contains_boolean(item) for item in value.to_numpy(dtype=object).flat)
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _coerce_integer_cell_id(value: object) -> int:
    """Return one validated integer cell identifier."""

    message = "cell_ids must contain integer identifiers"
    if isinstance(value, (bool, np.bool_)) or _is_missing_scalar(value):
        raise ValueError(message)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(message)
        try:
            integer = int(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if value != integer:
            raise ValueError(message)
        return integer
    if isinstance(value, str):
        text = value.strip()
        unsigned = text[1:] if text[:1] in {"+", "-"} else text
        if text.lower() in _MISSING_CELL_ID_STRINGS or not unsigned.isdigit():
            raise ValueError(message)
        return int(text)
    raise ValueError(message)


def _coerce_place_field_cell_ids(cell_ids: Any, *, n_cells: int) -> np.ndarray:
    """Return validated integer cell IDs for place-field diagnostics."""

    raw = np.asarray(cell_ids, dtype=object)
    if raw.ndim != 1 or raw.shape[0] != n_cells:
        raise ValueError("cell_ids must have one integer entry per cell")
    return np.asarray([_coerce_integer_cell_id(value) for value in raw], dtype=int)


def _coerce_place_field_numeric_arrays(
    rates_hz: Any,
    occupancy_s: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite nonnegative place-field rates and occupancies."""

    if _contains_boolean(rates_hz):
        raise ValueError("rates_hz must contain finite nonnegative values")
    if _contains_boolean(occupancy_s):
        raise ValueError("occupancy_s must contain finite nonnegative values")

    try:
        rates = np.asarray(rates_hz, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rates_hz must contain finite nonnegative values") from exc
    try:
        occupancy = np.asarray(occupancy_s, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("occupancy_s must contain finite nonnegative values") from exc

    if rates.ndim != 2:
        raise ValueError("rates_hz must have shape (n_cells, n_bins)")
    if occupancy.ndim != 1 or occupancy.shape[0] != rates.shape[1]:
        raise ValueError("occupancy_s must have one value per spatial bin")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("rates_hz must contain finite nonnegative values")
    if not np.all(np.isfinite(occupancy)) or np.any(occupancy < 0.0):
        raise ValueError("occupancy_s must contain finite nonnegative values")
    return rates, occupancy


def apply_advanced_result_place_field_cell_id_validation_patch() -> None:
    """Install validation for place-field quality inputs."""

    from . import advanced_result_diagnostics as diagnostics

    current = diagnostics.place_field_quality_from_arrays
    if getattr(current, _PATCHED_FLAG, False):
        return
    base = getattr(diagnostics, _BASE_ATTR, current)

    @wraps(base)
    def place_field_quality_from_arrays(rates_hz, occupancy_s, cell_ids=None):
        rates, occupancy = _coerce_place_field_numeric_arrays(rates_hz, occupancy_s)
        if cell_ids is not None:
            cell_ids = _coerce_place_field_cell_ids(cell_ids, n_cells=rates.shape[0])
        return base(rates, occupancy, cell_ids=cell_ids)

    setattr(place_field_quality_from_arrays, _PATCHED_FLAG, True)
    setattr(diagnostics, _BASE_ATTR, base)
    diagnostics.place_field_quality_from_arrays = place_field_quality_from_arrays


__all__ = ["apply_advanced_result_place_field_cell_id_validation_patch"]
