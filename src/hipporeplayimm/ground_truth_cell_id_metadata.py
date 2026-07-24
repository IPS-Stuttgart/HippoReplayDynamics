"""Strict parsing for saved train/test cell-ID metadata."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd


_PATCHED_FLAG = "_ground_truth_strict_cell_id_metadata_patch_applied"
_MISSING_TEXT_VALUES = frozenset({"", "nan", "na", "n/a", "none", "null", "<na>"})
_CELL_ID_METADATA_ERROR = "score-table cell IDs cell ID metadata must contain integer values"


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
    text = _cell_id_text(value).strip()
    if text.lower() in _MISSING_TEXT_VALUES:
        return None
    text = text.strip("[]()").replace(",", " ")
    if text.strip().lower() in _MISSING_TEXT_VALUES:
        return None
    return _integer_array_from_values(text.split())


_parse_cell_ids_strict.__name__ = "_parse_cell_ids"


def _integer_array_from_values(values: Any) -> np.ndarray:
    if isinstance(values, (list, tuple, set)):
        values = [
            bytes(value) if isinstance(value, (bytearray, memoryview)) else value
            for value in values
        ]
    parsed = [_parse_cell_id_value(value) for value in np.asarray(values, dtype=object).reshape(-1)]
    integer_info = np.iinfo(int)
    if any(value < integer_info.min or value > integer_info.max for value in parsed):
        raise ValueError(
            "score-table cell IDs cell ID metadata must fit the platform integer range"
        )
    return np.asarray(parsed, dtype=int)


def _cell_id_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray, memoryview, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(_CELL_ID_METADATA_ERROR) from exc
    return str(value)


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
            raise ValueError(_CELL_ID_METADATA_ERROR)
        return integer

    try:
        numeric = Decimal(_cell_id_text(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(_CELL_ID_METADATA_ERROR) from exc
    if not numeric.is_finite():
        raise ValueError("score-table cell IDs cell ID metadata must contain finite integer values")
    integer = numeric.to_integral_value()
    if numeric != integer:
        raise ValueError(_CELL_ID_METADATA_ERROR)
    return int(integer)


__all__ = ["apply_ground_truth_cell_id_metadata_patch"]
