"""Strict parsing for saved train/test cell-ID metadata."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd


_PATCHED_FLAG = "_ground_truth_strict_cell_id_metadata_patch_applied"
_MISSING_TEXT_VALUES = frozenset({"", "nan", "na", "n/a", "none", "null", "<na>"})


def apply_ground_truth_cell_id_metadata_patch() -> None:
    """Reject malformed cell-ID metadata instead of silently truncating it."""

    from . import ground_truth as gt

    if _ground_truth_cell_id_metadata_patch_current(gt):
        return
    gt._parse_cell_ids = _parse_cell_ids_strict
    setattr(gt, _PATCHED_FLAG, True)


def _ground_truth_cell_id_metadata_patch_current(gt: object) -> bool:
    return bool(
        getattr(gt, _PATCHED_FLAG, False)
        and getattr(gt, "_parse_cell_ids", None) is _parse_cell_ids_strict
    )


def _parse_cell_ids_strict(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return _integer_array_from_values(value)
    if isinstance(value, (list, tuple, set)):
        return _integer_array_from_values(list(value))
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in _MISSING_TEXT_VALUES:
        return None
    text = text.strip("[]()").replace(",", " ")
    if text.strip().lower() in _MISSING_TEXT_VALUES:
        return None
    return _integer_array_from_values(text.split())


_parse_cell_ids_strict.__name__ = "_parse_cell_ids"


def _integer_array_from_values(values: Any) -> np.ndarray:
    parsed = [_parse_cell_id_value(value) for value in np.asarray(values, dtype=object).reshape(-1)]
    return np.asarray(parsed, dtype=int)


def _parse_cell_id_value(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("score-table cell IDs cell ID metadata must not contain boolean identifiers")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("score-table cell IDs cell ID metadata must contain finite integer values")
        integer = int(value)
        if value != integer:
            raise ValueError("score-table cell IDs cell ID metadata must contain integer values")
        return integer

    try:
        if isinstance(value, (bytes, np.bytes_)):
            text = bytes(value).decode("utf-8")
        else:
            text = str(value)
        numeric = Decimal(text.strip())
    except (InvalidOperation, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("score-table cell IDs cell ID metadata must contain integer values") from exc
    if not numeric.is_finite():
        raise ValueError("score-table cell IDs cell ID metadata must contain finite integer values")
    integer = numeric.to_integral_value()
    if numeric != integer:
        raise ValueError("score-table cell IDs cell ID metadata must contain integer values")
    return int(integer)


__all__ = ["apply_ground_truth_cell_id_metadata_patch"]
