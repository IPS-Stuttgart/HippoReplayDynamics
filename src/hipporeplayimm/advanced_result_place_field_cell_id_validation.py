"""Validate place-field diagnostic cell identifiers before integer coercion."""

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


def _coerce_integer_cell_id(value: object) -> int:
    """Return one validated integer cell identifier."""

    message = "cell_ids must contain integer identifiers"
    if isinstance(value, (bool, np.bool_)) or _is_missing_scalar(value):
        raise ValueError(message)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        raise ValueError(message)
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


def apply_advanced_result_place_field_cell_id_validation_patch() -> None:
    """Install validation for place-field quality cell identifiers."""

    from . import advanced_result_diagnostics as diagnostics

    current = diagnostics.place_field_quality_from_arrays
    if getattr(current, _PATCHED_FLAG, False):
        return
    base = getattr(diagnostics, _BASE_ATTR, current)

    @wraps(base)
    def place_field_quality_from_arrays(rates_hz, occupancy_s, cell_ids=None):
        if cell_ids is not None:
            rates = np.asarray(rates_hz, dtype=float)
            if rates.ndim != 2:
                return base(rates_hz, occupancy_s, cell_ids=cell_ids)
            cell_ids = _coerce_place_field_cell_ids(cell_ids, n_cells=rates.shape[0])
        return base(rates_hz, occupancy_s, cell_ids=cell_ids)

    setattr(place_field_quality_from_arrays, _PATCHED_FLAG, True)
    setattr(diagnostics, _BASE_ATTR, base)
    diagnostics.place_field_quality_from_arrays = place_field_quality_from_arrays


__all__ = ["apply_advanced_result_place_field_cell_id_validation_patch"]
