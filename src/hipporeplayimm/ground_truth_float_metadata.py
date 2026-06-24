"""Strict numeric and boolean parsing for score-table metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_FLOAT_PATCHED_FLAG = "_ground_truth_strict_float_metadata_patch_applied"
_BOOL_PATCHED_FLAG = "_ground_truth_strict_bool_metadata_patch_applied"
_SCORE_BOOL_PATCHED_FLAG = "_score_metadata_strict_bool_metadata_patch_applied"
_EVIDENCE_BOOL_PATCHED_FLAG = "_evidence_reporting_strict_bool_metadata_patch_applied"


def apply_ground_truth_float_metadata_patch() -> None:
    """Reject malformed numeric/boolean metadata instead of propagating invalid configs."""

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

    from . import score_metadata as score_meta

    if not getattr(score_meta, _SCORE_BOOL_PATCHED_FLAG, False):

        def parse_score_bool(value: Any) -> bool:
            return _parse_bool_metadata_value("boolean metadata", value)

        parse_score_bool.__name__ = score_meta._parse_bool.__name__
        parse_score_bool.__doc__ = score_meta._parse_bool.__doc__
        score_meta._parse_bool = parse_score_bool
        setattr(score_meta, _SCORE_BOOL_PATCHED_FLAG, True)

    from . import evidence_reporting as evidence

    if not getattr(evidence, _EVIDENCE_BOOL_PATCHED_FLAG, False):

        def coerce_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
            def coerce(value: object) -> bool:
                return _parse_bool_metadata_value_or_default(value, default=default)

            return values.map(coerce).astype(bool)

        coerce_bool_series.__name__ = evidence._coerce_bool_series.__name__
        coerce_bool_series.__doc__ = evidence._coerce_bool_series.__doc__
        evidence._coerce_bool_series = coerce_bool_series

        # recovery_diagnostics_bool_patch imports the helper by value during package
        # initialisation; keep that alias synchronized before the patch is applied.
        from . import recovery_diagnostics_bool_patch

        recovery_diagnostics_bool_patch._coerce_bool_series = coerce_bool_series
        setattr(evidence, _EVIDENCE_BOOL_PATCHED_FLAG, True)


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


def _parse_bool_metadata_value_or_default(value: Any, *, default: bool) -> bool:
    try:
        return _parse_bool_metadata_value("boolean metadata", value)
    except ValueError:
        return bool(default)


__all__ = ["apply_ground_truth_float_metadata_patch"]