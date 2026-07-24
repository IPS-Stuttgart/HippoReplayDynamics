"""Strict score-table metadata parsing."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

_MISSING_METADATA_STRINGS = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_BYTE_BACKED_SCALARS = (bytes, bytearray, memoryview, np.bytes_)
_SCORE_METADATA_BOOL_PATCH_FLAG = "_score_metadata_bool_validation_patch_applied"
_CLUSTERLESS_STRING_PATCH_FLAG = "_score_metadata_string_missing_patch_applied"


def _bool_parse_error(value: object) -> ValueError:
    return ValueError(f"cannot parse boolean value {value!r}; boolean values must be true/false or binary 0/1")


def _numeric_parse_error(column: str) -> ValueError:
    return ValueError(f"{column} must contain finite numeric metadata")


def _unwrap_scalar_metadata(value: object, label: str = "metadata value") -> object:
    """Unwrap persisted singleton containers without accepting non-scalars."""

    seen: set[int] = set()
    while isinstance(value, (np.ndarray, list, tuple, np.generic)):
        if isinstance(value, np.ndarray):
            if value.size != 1 or id(value) in seen:
                raise ValueError(f"{label} must be scalar")
            seen.add(id(value))
            value = value.reshape(-1)[0]
        elif isinstance(value, (list, tuple)):
            if len(value) != 1 or id(value) in seen:
                raise ValueError(f"{label} must be scalar")
            seen.add(id(value))
            value = value[0]
        else:
            value = value.item()
    return value


def _parse_strict_bool(value: object) -> bool:
    """Parse boolean-like metadata without accepting arbitrary numerics."""

    value = _unwrap_scalar_metadata(value, "boolean metadata")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        if integer == 0:
            return False
        if integer == 1:
            return True
        raise _bool_parse_error(value)
    if isinstance(value, (float, np.floating)):
        return _parse_numeric_bool(float(value), value)

    text = _metadata_text_or_none(value)
    if text is None:
        raise _bool_parse_error(value)
    text = text.lower()
    if text in {"true", "yes", "on"}:
        return True
    if text in {"false", "no", "off"}:
        return False
    try:
        numeric = float(text)
    except ValueError as exc:
        raise _bool_parse_error(value) from exc
    return _parse_numeric_bool(numeric, value)


def _parse_numeric_bool(numeric: float, original: object) -> bool:
    if not np.isfinite(numeric):
        raise _bool_parse_error(original)
    if np.isclose(numeric, 0.0, rtol=0.0, atol=0.0):
        return False
    if np.isclose(numeric, 1.0, rtol=0.0, atol=0.0):
        return True
    raise _bool_parse_error(original)


def _metadata_text_or_none(value: object) -> str | None:
    value = _unwrap_scalar_metadata(value)
    if isinstance(value, _BYTE_BACKED_SCALARS):
        text = bytes(value).decode("utf-8", errors="replace").strip()
    else:
        text = str(value).strip()
    if text.lower() in _MISSING_METADATA_STRINGS:
        return None
    return text or None


def _metadata_float_from_value(value: object, column: str) -> float | None:
    value = _unwrap_scalar_metadata(value, column)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        raise _numeric_parse_error(column)

    text = _metadata_text_or_none(value)
    if text is None:
        return None
    try:
        numeric = float(text)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _numeric_parse_error(column) from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{column} must be finite")
    return float(numeric)


def _metadata_int_from_value(value: object, column: str) -> int | None:
    """Parse integer metadata without losing precision through binary64."""

    value = _unwrap_scalar_metadata(value, column)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        raise _numeric_parse_error(column)
    if isinstance(value, (int, np.integer)):
        return int(value)

    text = _metadata_text_or_none(value)
    if text is None:
        return None
    try:
        numeric = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise _numeric_parse_error(column) from exc
    if not numeric.is_finite():
        raise ValueError(f"{column} must be finite")

    integer = numeric.to_integral_value()
    if abs(numeric - integer) > Decimal("1e-9"):
        raise ValueError(f"{column} must be an integer")
    return int(integer)


def _unique_int_from_columns(frame: pd.DataFrame, columns: tuple[str, ...], default: int) -> int:
    """Return one exact integer shared by the requested metadata columns."""

    values: list[int] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column]:
            parsed = _metadata_int_from_value(value, column)
            if parsed is not None:
                values.append(parsed)
    if not values:
        return int(default)
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return int(first)


def _unique_string_from_columns(frame: pd.DataFrame, columns: tuple[str, ...], default: str) -> str:
    values: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            text = _metadata_text_or_none(value)
            if text is not None:
                values.append(text)
    if not values:
        return str(default)
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return first


def _unique_bool_from_column(frame: pd.DataFrame, column: str, default: bool) -> bool:
    values: list[bool] = []
    if column in frame.columns:
        for value in frame[column].dropna():
            if _metadata_text_or_none(value) is None:
                continue
            values.append(_parse_strict_bool(value))
    if not values:
        return bool(default)
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{column} contains multiple values")
    return bool(first)


def _optional_float_from_columns(frame: pd.DataFrame, columns: tuple[str, ...], default: float | None) -> float | None:
    values: list[float] = []
    joined_columns = " / ".join(columns)
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            value = _unwrap_scalar_metadata(value, joined_columns)
            if isinstance(value, (bool, np.bool_)):
                raise _numeric_parse_error(joined_columns)
            text = _metadata_text_or_none(value)
            if text is None:
                continue
            try:
                numeric = float(text)
            except (TypeError, ValueError, OverflowError) as exc:
                raise _numeric_parse_error(joined_columns) from exc
            if not np.isfinite(numeric):
                raise ValueError(f"{joined_columns} must be finite")
            values.append(float(numeric))
    if not values:
        return default
    first = values[0]
    if any(not np.isclose(value, first) for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return float(first)


def _score_metadata_patch_current(score_metadata_module: object) -> bool:
    return bool(
        getattr(score_metadata_module, _SCORE_METADATA_BOOL_PATCH_FLAG, False)
        and getattr(score_metadata_module, "_parse_bool", None) is _parse_strict_bool
        and getattr(score_metadata_module, "_unique_bool_from_column", None) is _unique_bool_from_column
        and getattr(score_metadata_module, "_unique_string_from_columns", None) is _unique_string_from_columns
        and getattr(score_metadata_module, "_metadata_float_from_value", None) is _metadata_float_from_value
        and getattr(score_metadata_module, "_unique_int_from_columns", None) is _unique_int_from_columns
    )


def _clusterless_string_patch_current(clusterless_ground_truth_module: object) -> bool:
    return bool(
        getattr(clusterless_ground_truth_module, _CLUSTERLESS_STRING_PATCH_FLAG, False)
        and getattr(clusterless_ground_truth_module, "_unique_string_from_columns", None) is _unique_string_from_columns
        and getattr(clusterless_ground_truth_module, "_optional_float_from_columns", None) is _optional_float_from_columns
    )


def apply_score_metadata_bool_validation_patch() -> None:
    """Install strict parsing and string metadata handling."""

    from . import score_metadata as score_metadata_module

    if not _score_metadata_patch_current(score_metadata_module):
        score_metadata_module._parse_bool = _parse_strict_bool
        score_metadata_module._unique_bool_from_column = _unique_bool_from_column
        score_metadata_module._unique_string_from_columns = _unique_string_from_columns
        score_metadata_module._metadata_float_from_value = _metadata_float_from_value
        score_metadata_module._unique_int_from_columns = _unique_int_from_columns
        setattr(score_metadata_module, _SCORE_METADATA_BOOL_PATCH_FLAG, True)

    try:
        from . import clusterless_ground_truth as clusterless_ground_truth_module
    except ImportError:
        return
    if _clusterless_string_patch_current(clusterless_ground_truth_module):
        return
    clusterless_ground_truth_module._unique_string_from_columns = _unique_string_from_columns
    clusterless_ground_truth_module._optional_float_from_columns = _optional_float_from_columns
    setattr(clusterless_ground_truth_module, _CLUSTERLESS_STRING_PATCH_FLAG, True)
