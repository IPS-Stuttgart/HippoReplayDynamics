"""Strict integer parsing for post-hoc ground-truth score metadata."""

from __future__ import annotations

from typing import Any

import numpy as np


_PATCHED_FLAG = "_ground_truth_strict_integer_metadata_patch_applied"


def apply_ground_truth_integer_metadata_patch() -> None:
    """Reject fractional integer metadata instead of silently truncating it."""

    from . import ground_truth as gt

    if getattr(gt, _PATCHED_FLAG, False):
        return

    def unique_int_from_column(frame: Any, column: str, default: int) -> int:
        values = [
            _parse_integer_metadata_value(column, value)
            for value in gt._iter_present_column_values(frame, (column,))
        ]
        if not values:
            return int(default)
        first = values[0]
        if any(value != first for value in values[1:]):
            raise ValueError(f"{column} contains multiple values")
        return int(first)

    def parse_cell_ids(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return _parse_cell_id_values(value.reshape(-1))
        if isinstance(value, (list, tuple, set)):
            return _parse_cell_id_values(list(value))
        if gt._is_missing_scalar(value):
            return None
        text = str(value).strip()
        missing_values = getattr(gt, "_MISSING_TEXT_VALUES", frozenset({"", "nan"}))
        if text.lower() in missing_values:
            return None
        text = text.strip("[]()").replace(",", " ")
        if not text:
            return np.array([], dtype=int)
        return _parse_cell_id_values(text.split())

    unique_int_from_column.__name__ = gt._unique_int_from_column.__name__
    unique_int_from_column.__doc__ = gt._unique_int_from_column.__doc__
    parse_cell_ids.__name__ = gt._parse_cell_ids.__name__
    parse_cell_ids.__doc__ = gt._parse_cell_ids.__doc__
    gt._unique_int_from_column = unique_int_from_column
    gt._parse_cell_ids = parse_cell_ids
    setattr(gt, _PATCHED_FLAG, True)


def _parse_integer_metadata_value(column: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{column} must contain integer values")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain integer values") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{column} must contain finite integer values")
    integer = int(round(numeric))
    if not np.isclose(numeric, integer, rtol=0.0, atol=1e-9):
        raise ValueError(f"{column} must contain integer values")
    return int(integer)


def _parse_cell_id_values(values: Any) -> np.ndarray:
    return np.asarray(
        [
            _parse_integer_metadata_value("score-table cell IDs", value)
            for value in np.asarray(values, dtype=object).reshape(-1)
        ],
        dtype=int,
    )


__all__ = ["apply_ground_truth_integer_metadata_patch"]
