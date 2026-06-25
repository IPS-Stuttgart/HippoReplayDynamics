"""Strict score-table metadata parsing."""

from __future__ import annotations

import numpy as np
import pandas as pd

_MISSING_METADATA_STRINGS = {"", "nan", "na", "n/a", "none", "<na>"}
_CLUSTERLESS_STRING_PATCH_FLAG = "_score_metadata_string_missing_patch_applied"


def _parse_strict_bool(value: object) -> bool:
    """Parse boolean-like metadata without accepting arbitrary numerics."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return _parse_numeric_bool(float(value), value)

    text = str(value).strip().lower()
    if text in {"true", "yes", "on"}:
        return True
    if text in {"false", "no", "off"}:
        return False
    try:
        numeric = float(text)
    except ValueError:
        pass
    else:
        return _parse_numeric_bool(numeric, value)
    raise ValueError(f"cannot parse boolean value {value!r}")


def _parse_numeric_bool(numeric: float, original: object) -> bool:
    if not np.isfinite(numeric):
        raise ValueError(f"cannot parse boolean value {original!r}")
    if np.isclose(numeric, 0.0, rtol=0.0, atol=0.0):
        return False
    if np.isclose(numeric, 1.0, rtol=0.0, atol=0.0):
        return True
    raise ValueError(f"cannot parse boolean value {original!r}")


def _metadata_text_or_none(value: object) -> str | None:
    text = str(value).strip()
    if text.lower() in _MISSING_METADATA_STRINGS:
        return None
    return text or None


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


def _optional_float_from_columns(frame: pd.DataFrame, columns: tuple[str, ...], default: float | None) -> float | None:
    values: list[float] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            text = _metadata_text_or_none(value)
            if text is None:
                continue
            numeric = float(text)
            if not np.isfinite(numeric):
                raise ValueError(f"{' / '.join(columns)} must be finite")
            values.append(float(numeric))
    if not values:
        return default
    first = values[0]
    if any(not np.isclose(value, first) for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return float(first)


def apply_score_metadata_bool_validation_patch() -> None:
    """Install strict parsing and string metadata handling."""

    from . import score_metadata as score_metadata_module

    if not getattr(score_metadata_module, "_score_metadata_bool_validation_patch_applied", False):
        score_metadata_module._parse_bool = _parse_strict_bool
        score_metadata_module._unique_string_from_columns = _unique_string_from_columns
        score_metadata_module._score_metadata_bool_validation_patch_applied = True

    try:
        from . import clusterless_ground_truth as clusterless_ground_truth_module
    except ImportError:
        return
    if getattr(clusterless_ground_truth_module, _CLUSTERLESS_STRING_PATCH_FLAG, False):
        return
    clusterless_ground_truth_module._unique_string_from_columns = _unique_string_from_columns
    clusterless_ground_truth_module._optional_float_from_columns = _optional_float_from_columns
    setattr(clusterless_ground_truth_module, _CLUSTERLESS_STRING_PATCH_FLAG, True)
