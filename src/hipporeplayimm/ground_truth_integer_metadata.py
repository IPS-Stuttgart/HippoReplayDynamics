"""Strict scalar parsing for post-hoc ground-truth score metadata."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_PATCHED_FLAG = "_ground_truth_strict_integer_metadata_patch_applied"
_UNIQUE_INT_WRAPPER_MARKER = "_hipporeplayimm_ground_truth_integer_metadata_unique_int"
_PARSE_CELL_IDS_WRAPPER_MARKER = "_hipporeplayimm_ground_truth_integer_metadata_parse_cell_ids"
_WINDOW_FLOAT_WRAPPER_MARKER = "_hipporeplayimm_ground_truth_window_unique_float"
_CELL_SPLIT_STRATA_WRAPPER_MARKER = (
    "_hipporeplayimm_ground_truth_integer_metadata_cell_split_strata"
)
_COMPARE_SCORES_WRAPPER_MARKER = (
    "_hipporeplayimm_ground_truth_integer_metadata_compare_scores_csv"
)
_CELL_ID_PATCHED_FLAG = "_ground_truth_strict_cell_id_metadata_patch_applied"
_MISSING_TEXT_VALUES = frozenset({"", "na", "n/a", "nan", "none", "null", "<na>"})
_GROUND_TRUTH_INTEGER_SCORE_COLUMNS = (
    "simulation_random_seed",
    "random_seed",
    "benchmark_random_seed",
    "null_random_seed",
    "benchmark_event_subset_seed",
    "simulation_event_index",
    "event_index",
    "window_index",
    "null_index",
    "matched_null_rank",
    "template_event_index",
    "benchmark_cell_split_index",
    "cell_split_index",
    "benchmark_cell_split_seed",
    "cell_split_seed",
    "benchmark_cell_split_strata",
    "cell_split_count",
    "cell_split_shard_index",
    "cell_split_shard_count",
    "split_shard_index",
    "split_shard_count",
)


def apply_ground_truth_integer_metadata_patch() -> None:
    """Reject malformed scalar metadata instead of silently coercing it."""

    from . import ground_truth as gt

    _patch_compare_scores_csv_read(gt)
    _patch_window_float_metadata()
    _patch_cell_split_strata_metadata()

    if not _unique_int_patch_current(gt):

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
        setattr(unique_int_from_column, _UNIQUE_INT_WRAPPER_MARKER, True)
        gt._unique_int_from_column = unique_int_from_column

    if not _parse_cell_ids_patch_current(gt) and not _cell_id_metadata_patch_current(gt):

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

        parse_cell_ids.__name__ = gt._parse_cell_ids.__name__
        parse_cell_ids.__doc__ = gt._parse_cell_ids.__doc__
        setattr(parse_cell_ids, _PARSE_CELL_IDS_WRAPPER_MARKER, True)
        gt._parse_cell_ids = parse_cell_ids

    setattr(gt, _PATCHED_FLAG, True)


def _patch_compare_scores_csv_read(gt: Any) -> None:
    """Preserve exact integer identity columns before ground-truth decoding."""

    current = gt.compare_scores_to_ground_truth
    if getattr(current, _COMPARE_SCORES_WRAPPER_MARKER, False):
        return

    @wraps(current)
    def compare_scores_to_ground_truth_exact_integer_csv(
        root: str | Path,
        scores: str | Path | pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not isinstance(scores, pd.DataFrame):
            scores = _read_ground_truth_score_csv(scores)
        return current(root, scores, *args, **kwargs)

    setattr(
        compare_scores_to_ground_truth_exact_integer_csv,
        _COMPARE_SCORES_WRAPPER_MARKER,
        True,
    )
    gt.compare_scores_to_ground_truth = compare_scores_to_ground_truth_exact_integer_csv


def _read_ground_truth_score_csv(path: str | Path) -> pd.DataFrame:
    """Read score CSVs without float-rounding integer event/scope identifiers."""

    frame = pd.read_csv(
        path,
        dtype={column: "string" for column in _GROUND_TRUTH_INTEGER_SCORE_COLUMNS},
    )
    for column in _GROUND_TRUTH_INTEGER_SCORE_COLUMNS:
        if column not in frame.columns:
            continue
        frame[column] = pd.Series(
            [
                (
                    pd.NA
                    if _is_missing_integer_metadata_value(value)
                    else _parse_integer_metadata_value(column, value)
                )
                for value in frame[column]
            ],
            index=frame.index,
            dtype=object,
        )
    return frame


def _is_missing_integer_metadata_value(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return True
    return isinstance(value, str) and value.strip().lower() in _MISSING_TEXT_VALUES


def _patch_window_float_metadata() -> None:
    """Make saved replay-window floats strict without changing CSV numeric text support."""

    from . import ground_truth_window_scope as window_scope

    current = window_scope._unique_finite_float
    if getattr(current, _WINDOW_FLOAT_WRAPPER_MARKER, False):
        return

    def unique_finite_float(frame: Any, column: str) -> float | None:
        if column not in frame.columns:
            return None
        values: list[float] = []
        for value in frame[column]:
            if window_scope._is_missing_scalar(value):
                continue
            text = str(value).strip()
            if text.lower() in window_scope._MISSING_TEXT_VALUES:
                continue
            values.append(_parse_finite_float_metadata_value(column, value))
        if not values:
            return None
        first = values[0]
        if any(not np.isclose(value, first, rtol=0.0, atol=1e-12) for value in values[1:]):
            raise ValueError(f"{column} contains multiple values within one replay-window decode group")
        return float(first)

    unique_finite_float.__name__ = current.__name__
    unique_finite_float.__doc__ = current.__doc__
    setattr(unique_finite_float, _WINDOW_FLOAT_WRAPPER_MARKER, True)
    window_scope._unique_finite_float = unique_finite_float


def _patch_cell_split_strata_metadata() -> None:
    """Require mathematically integral saved cell-split strata metadata."""

    from . import benchmark_cell_split_metadata as cell_split_metadata

    current = cell_split_metadata._cell_split_strata_from_scores
    if getattr(current, _CELL_SPLIT_STRATA_WRAPPER_MARKER, False):
        return

    def cell_split_strata_from_scores(frame: Any, default: int) -> int:
        column = "benchmark_cell_split_strata"
        values = [
            _parse_integer_metadata_value(column, value)
            for value in cell_split_metadata._string_metadata_values(frame, column)
        ]
        if not values:
            return int(default)
        first = values[0]
        if any(value != first for value in values[1:]):
            raise ValueError(f"{column} contains multiple values")
        return int(first)

    cell_split_strata_from_scores.__name__ = current.__name__
    cell_split_strata_from_scores.__doc__ = current.__doc__
    setattr(
        cell_split_strata_from_scores,
        _CELL_SPLIT_STRATA_WRAPPER_MARKER,
        True,
    )
    cell_split_metadata._cell_split_strata_from_scores = cell_split_strata_from_scores


def _unique_int_patch_current(gt: object) -> bool:
    return bool(getattr(getattr(gt, "_unique_int_from_column", None), _UNIQUE_INT_WRAPPER_MARKER, False))


def _parse_cell_ids_patch_current(gt: object) -> bool:
    return bool(getattr(getattr(gt, "_parse_cell_ids", None), _PARSE_CELL_IDS_WRAPPER_MARKER, False))


def _cell_id_metadata_patch_current(gt: object) -> bool:
    from .ground_truth_cell_id_metadata import _ground_truth_cell_id_metadata_patch_current

    return bool(_ground_truth_cell_id_metadata_patch_current(gt))


def _parse_integer_metadata_value(column: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{column} must contain integer values")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{column} must contain finite integer values")
        integer = int(value)
        if value != integer:
            raise ValueError(f"{column} must contain integer values")
        return integer

    try:
        numeric = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{column} must contain integer values") from exc
    if not numeric.is_finite():
        raise ValueError(f"{column} must contain finite integer values")
    integer = numeric.to_integral_value()
    if numeric != integer:
        raise ValueError(f"{column} must contain integer values")
    return int(integer)


def _unwrap_finite_float_metadata_scalar(column: str, value: Any) -> Any:
    """Unwrap nested zero-dimensional object arrays without scalar coercion."""

    current = value
    seen: set[int] = set()
    while True:
        if isinstance(current, (bool, np.bool_, complex, np.complexfloating)):
            raise ValueError(f"{column} must contain finite numeric scalar values")
        try:
            raw = np.asarray(current)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain finite numeric scalar values") from exc
        if raw.ndim != 0:
            raise ValueError(f"{column} must contain finite numeric scalar values")
        item = raw.item()
        if raw.dtype != object or not isinstance(current, np.ndarray):
            return item
        marker = id(current)
        if marker in seen or item is current:
            raise ValueError(f"{column} must contain finite numeric scalar values")
        seen.add(marker)
        current = item


def _parse_finite_float_metadata_value(column: str, value: Any) -> float:
    item = _unwrap_finite_float_metadata_scalar(column, value)
    if isinstance(item, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(f"{column} must contain finite numeric scalar values")
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{column} must contain finite numeric scalar values") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{column} must contain finite numeric scalar values")
    return float(numeric)


def _parse_cell_id_values(values: Any) -> np.ndarray:
    parsed = np.asarray(
        [
            _parse_integer_metadata_value("score-table cell IDs", value)
            for value in np.asarray(values, dtype=object).reshape(-1)
        ],
        dtype=int,
    )
    return np.sort(parsed)


__all__ = ["apply_ground_truth_integer_metadata_patch"]
