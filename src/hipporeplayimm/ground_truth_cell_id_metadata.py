"""Strict parsing for cell IDs and behavioral well-sequence metadata."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


_PATCHED_FLAG = "_ground_truth_strict_cell_id_metadata_patch_applied"
_MISSING_TEXT_VALUES = frozenset({"", "nan", "na", "n/a", "none", "null", "<na>"})
_CELL_ID_METADATA_ERROR = "score-table cell IDs cell ID metadata must contain integer values"
_ACTIVE_GOAL_PATCHED_FLAG = "_ground_truth_active_goal_combined_validation_patch_applied"
_FLOAT_HELPER_PATCHED_FLAG = "_ground_truth_active_goal_repair_wrapper"
_NUMERIC_SCALAR_PATCHED_FLAG = "_ground_truth_nested_numeric_scalar_validation_patch_applied"
_WELL_SEQUENCE_HELPER_PATCHED_FLAG = "_ground_truth_well_sequence_time_validation_patch_applied"
_WELL_SEQUENCE_TIME_ERROR = "well sequence fill times must contain finite numeric values"
_LEGACY_ACTIVE_GOAL_MARKERS = (
    "_ground_truth_direct_active_goal_numeric_validation_patch_applied",
    "_ground_truth_direct_active_goal_well_id_validation_patch_applied",
)
_ACTIVE_GOAL_WRAPPER_CODE: Any | None = None


def apply_ground_truth_cell_id_metadata_patch() -> None:
    """Reject malformed cell-ID and well-sequence metadata."""

    from . import ground_truth as gt
    from . import ground_truth_float_metadata as float_metadata

    _patch_float_metadata_numeric_scalar_helper(float_metadata)
    _patch_float_metadata_well_sequence_helper(float_metadata)
    _patch_float_metadata_active_goal_helper(float_metadata)
    _install_active_goal_validation(gt, float_metadata)

    if _ground_truth_cell_id_metadata_patch_current(gt):
        return
    gt._parse_cell_ids = _parse_cell_ids_strict
    setattr(gt, _PATCHED_FLAG, True)


def _patch_float_metadata_numeric_scalar_helper(float_metadata: Any) -> None:
    """Reject malformed values hidden in nested zero-dimensional object wrappers."""

    current = float_metadata._parse_config_scalar
    if getattr(current, _NUMERIC_SCALAR_PATCHED_FLAG, False):
        return

    @wraps(current)
    def parse_config_scalar(name: str, value: Any) -> float:
        candidate = value
        seen: set[int] = set()
        while True:
            if isinstance(candidate, (bool, np.bool_)):
                raise TypeError(f"{name} must be numeric, not boolean")
            try:
                raw = np.asarray(candidate)
            except (TypeError, ValueError):
                return current(name, candidate)
            if raw.ndim != 0:
                return current(name, candidate)
            if raw.dtype != object:
                return current(name, candidate)
            try:
                item = raw.item()
            except ValueError:
                return current(name, candidate)
            marker = id(item)
            if item is candidate or marker in seen:
                raise TypeError(f"{name} must be a scalar")
            seen.add(marker)
            candidate = item

    setattr(parse_config_scalar, _NUMERIC_SCALAR_PATCHED_FLAG, True)
    setattr(parse_config_scalar, "__hipporeplayimm_original__", current)
    float_metadata._parse_config_scalar = parse_config_scalar


def _patch_float_metadata_well_sequence_helper(float_metadata: Any) -> None:
    """Share well-sequence time validation across all ground-truth helpers."""

    current = float_metadata._validate_well_sequence_ids
    if getattr(current, _WELL_SEQUENCE_HELPER_PATCHED_FLAG, False):
        return

    @wraps(current)
    def validate_well_sequence(well_sequence: Any) -> None:
        current(well_sequence)
        _validate_well_sequence_time_order(well_sequence)

    setattr(validate_well_sequence, _WELL_SEQUENCE_HELPER_PATCHED_FLAG, True)
    setattr(validate_well_sequence, "__hipporeplayimm_original__", current)
    float_metadata._validate_well_sequence_ids = validate_well_sequence


def _patch_float_metadata_active_goal_helper(float_metadata: Any) -> None:
    """Repair active-goal validation after every legacy float-helper refresh."""

    current = float_metadata._patch_direct_ground_truth_numeric_helpers
    if getattr(current, _FLOAT_HELPER_PATCHED_FLAG, False):
        return

    @wraps(current)
    def patch_direct_ground_truth_numeric_helpers(gt: Any) -> None:
        current(gt)
        _patch_float_metadata_well_sequence_helper(float_metadata)
        _install_active_goal_validation(gt, float_metadata)

    setattr(patch_direct_ground_truth_numeric_helpers, _FLOAT_HELPER_PATCHED_FLAG, True)
    float_metadata._patch_direct_ground_truth_numeric_helpers = patch_direct_ground_truth_numeric_helpers


def _install_active_goal_validation(gt: Any, float_metadata: Any) -> None:
    """Install one non-recursive wrapper for timestamp and well-sequence validation."""

    current = gt.active_goal_at_time
    if _active_goal_patch_current(current):
        return

    base = _unwrap_legacy_active_goal(current)
    if _active_goal_patch_current(base):
        gt.active_goal_at_time = base
        return

    @wraps(base)
    def active_goal_at_time(session: Any, time_s: float) -> int | None:
        time_value = float_metadata._parse_config_scalar("time_s", time_s)
        float_metadata._validate_well_sequence_ids(session.well_sequence)
        return base(session, time_value)

    global _ACTIVE_GOAL_WRAPPER_CODE
    _ACTIVE_GOAL_WRAPPER_CODE = active_goal_at_time.__code__
    setattr(active_goal_at_time, _ACTIVE_GOAL_PATCHED_FLAG, True)
    gt.active_goal_at_time = active_goal_at_time


def _active_goal_patch_current(function: Any) -> bool:
    """Reject legacy wrappers that copied this patch's marker via ``wraps``."""

    return bool(
        getattr(function, _ACTIVE_GOAL_PATCHED_FLAG, False)
        and _ACTIVE_GOAL_WRAPPER_CODE is not None
        and getattr(function, "__code__", None) is _ACTIVE_GOAL_WRAPPER_CODE
    )


def _unwrap_legacy_active_goal(function: Any) -> Any:
    """Remove only the two legacy wrappers that were accidentally self-linked."""

    current = function
    seen: set[int] = set()
    while id(current) not in seen and any(
        getattr(current, marker, False) for marker in _LEGACY_ACTIVE_GOAL_MARKERS
    ):
        seen.add(id(current))
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None or wrapped is current:
            break
        current = wrapped
    return current


def _validate_well_sequence_time_order(well_sequence: Any) -> None:
    """Reject malformed or non-chronological fill times before they are consumed."""

    if well_sequence is None:
        return
    raw = np.asarray(well_sequence)
    if raw.size == 0 or raw.ndim != 2 or raw.shape[1] < 2:
        return

    if np.issubdtype(raw.dtype, np.bool_) or np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(_WELL_SEQUENCE_TIME_ERROR)

    if np.issubdtype(raw.dtype, np.number):
        times = raw[:, 0]
        if not np.all(np.isfinite(times)):
            raise ValueError(_WELL_SEQUENCE_TIME_ERROR)
    else:
        object_times = np.asarray(well_sequence, dtype=object)[:, 0]
        times = np.asarray(
            [_finite_real_well_fill_time(value) for value in object_times],
            dtype=np.longdouble,
        )

    if np.any(times[1:] < times[:-1]):
        raise ValueError("well sequence fill times must be nondecreasing")


def _finite_real_well_fill_time(value: Any) -> np.longdouble:
    """Return one finite real fill timestamp without lossy complex coercion."""

    current = value
    seen_arrays: set[int] = set()
    while isinstance(current, np.ndarray):
        if current.ndim != 0:
            raise ValueError(_WELL_SEQUENCE_TIME_ERROR)
        marker = id(current)
        if marker in seen_arrays:
            raise ValueError(_WELL_SEQUENCE_TIME_ERROR)
        seen_arrays.add(marker)
        try:
            item = current.item()
        except (TypeError, ValueError) as exc:
            raise ValueError(_WELL_SEQUENCE_TIME_ERROR) from exc
        if item is current:
            raise ValueError(_WELL_SEQUENCE_TIME_ERROR)
        current = item

    if isinstance(
        current,
        (
            bool,
            np.bool_,
            complex,
            np.complexfloating,
            str,
            bytes,
            bytearray,
            memoryview,
            np.str_,
            np.bytes_,
        ),
    ):
        raise ValueError(_WELL_SEQUENCE_TIME_ERROR)

    try:
        numeric = np.longdouble(current)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_WELL_SEQUENCE_TIME_ERROR) from exc
    if not np.isfinite(numeric):
        raise ValueError(_WELL_SEQUENCE_TIME_ERROR)
    return numeric


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
