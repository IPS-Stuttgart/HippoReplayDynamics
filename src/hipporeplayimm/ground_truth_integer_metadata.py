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

    unique_int_from_column.__name__ = gt._unique_int_from_column.__name__
    unique_int_from_column.__doc__ = gt._unique_int_from_column.__doc__
    gt._unique_int_from_column = unique_int_from_column
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


__all__ = ["apply_ground_truth_integer_metadata_patch"]
