"""Strict numeric parsing for post-hoc ground-truth score metadata."""

from __future__ import annotations

from typing import Any

import numpy as np

_FLOAT_PATCHED_FLAG = "_ground_truth_strict_float_metadata_patch_applied"
_BOOL_PATCHED_FLAG = "_ground_truth_strict_bool_metadata_patch_applied"


def apply_ground_truth_float_metadata_patch() -> None:
    """Reject malformed numeric metadata instead of propagating invalid configs."""

    from . import ground_truth as gt

    if not getattr(gt, _FLOAT_PATCHED_FLAG, False):

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
        setattr(gt, _FLOAT_PATCHED_FLAG, True)

    if not getattr(gt, _BOOL_PATCHED_FLAG, False):

        def unique_bool_from_column(frame: Any, column: str, default: bool) -> bool:
            values = [
                _parse_bool_metadata_value(column, value)
                for value in gt._iter_present_column_values(frame, (column,))
            ]
            if not values:
                return bool(default)
            first = values[0]
            if any(value != first for value in values[1:]):
                raise ValueError(f"{column} contains multiple values")
            return bool(first)

        def parse_bool(value: Any) -> bool:
            return _parse_bool_metadata_value("boolean metadata", value)

        unique_bool_from_column.__name__ = gt._unique_bool_from_column.__name__
        unique_bool_from_column.__doc__ = gt._unique_bool_from_column.__doc__
        parse_bool.__name__ = gt._parse_bool.__name__
        parse_bool.__doc__ = gt._parse_bool.__doc__
        gt._unique_bool_from_column = unique_bool_from_column
        gt._parse_bool = parse_bool
        setattr(gt, _BOOL_PATCHED_FLAG, True)


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


def _parse_bool_metadata_value(column: str, value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        raise ValueError(f"{column} must contain boolean values")
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "off"}:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must contain boolean values") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"cannot parse boolean value for {column}")
    if np.isclose(numeric, 0.0, rtol=0.0, atol=0.0):
        return False
    if np.isclose(numeric, 1.0, rtol=0.0, atol=0.0):
        return True
    raise ValueError(f"{column} must contain boolean values")


__all__ = ["apply_ground_truth_float_metadata_patch"]
