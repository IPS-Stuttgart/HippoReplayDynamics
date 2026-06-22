"""Strict float parsing for post-hoc ground-truth score metadata."""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCHED_FLAG = "_ground_truth_strict_float_metadata_patch_applied"


def apply_ground_truth_float_metadata_patch() -> None:
    """Reject non-finite float metadata instead of propagating invalid configs."""

    from . import ground_truth as gt

    if getattr(gt, _PATCHED_FLAG, False):
        return

    def unique_float_from_columns(frame: Any, columns: tuple[str, ...], default: float) -> float:
        values = [
            _parse_float_metadata_value(" / ".join(columns), value)
            for value in gt._iter_present_column_values(frame, columns)
        ]
        if not values:
            return float(default)
        first = values[0]
        if any(not np.isclose(value, first, rtol=1e-05, atol=1e-08) for value in values[1:]):
            raise ValueError(f"{' / '.join(columns)} contains multiple values")
        return float(first)

    def unique_optional_float_from_column(frame: Any, column: str, default: float | None) -> float | None:
        values = [
            _parse_float_metadata_value(column, value)
            for value in gt._iter_present_column_values(frame, (column,))
        ]
        if not values:
            return default
        first = values[0]
        if any(not np.isclose(value, first, rtol=1e-05, atol=1e-08) for value in values[1:]):
            raise ValueError(f"{column} contains multiple values")
        return float(first)

    unique_float_from_columns.__name__ = gt._unique_float_from_columns.__name__
    unique_float_from_columns.__doc__ = gt._unique_float_from_columns.__doc__
    unique_optional_float_from_column.__name__ = gt._unique_optional_float_from_column.__name__
    unique_optional_float_from_column.__doc__ = gt._unique_optional_float_from_column.__doc__
    gt._unique_float_from_columns = unique_float_from_columns
    gt._unique_optional_float_from_column = unique_optional_float_from_column
    setattr(gt, _PATCHED_FLAG, True)


def _parse_float_metadata_value(column: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{column} must contain finite numeric values")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain finite numeric values") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{column} must contain finite numeric values")
    return float(numeric)


__all__ = ["apply_ground_truth_float_metadata_patch"]
