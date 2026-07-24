"""Robust scalar parsing and event scoping for recovery diagnostics."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from operator import index as exact_index
from typing import Any

import numpy as np
import pandas as pd

from .evidence_reporting import _coerce_bool_series

_PATCHED_FLAG = "_recovery_diagnostics_bool_scalar_patch_applied"
_BOOL_FLAG = "_recovery_diagnostics_bool_coerce_bool_wrapper"
_ROW_BOOL_FLAG = "_recovery_diagnostics_bool_row_bool_wrapper"
_BOOL_SERIES_FLAG = "_recovery_diagnostics_bool_series_wrapper"
_FLOAT_FLAG = "_recovery_diagnostics_bool_coerce_float_wrapper"
_ROW_FLOAT_FLAG = "_recovery_diagnostics_bool_row_float_wrapper"
_EVENT_INDEX_FLAG = "_recovery_diagnostics_exact_event_index_wrapper"
_SUCCESS_FLAG = "_recovery_diagnostics_bool_successful_finite_scores_wrapper"
_COMPARABLE_FLAG = "_recovery_diagnostics_bool_comparable_mask_wrapper"
_EVENT_DIAGNOSTICS_FLAG = "_recovery_diagnostics_scoped_event_diagnostics_wrapper"
_CERTIFIED_EVENT_FLAG = "_recovery_diagnostics_scoped_certified_event_wrapper"
_CERTIFIED_SUMMARY_FLAG = "_recovery_diagnostics_scoped_certified_summary_wrapper"
_HELPERS = {
    "_coerce_bool": (_coerce_bool := lambda value, default=False: _bool_value(value, default), _BOOL_FLAG),
    "_row_bool": (_row_bool := lambda row, column, default: _bool_value(row[column], default) if column in row.index else bool(default), _ROW_BOOL_FLAG),
    "_coerce_bool_series": (_bool_series := lambda values, default=False: _map_bool(values, default), _BOOL_SERIES_FLAG),
    "_coerce_float": (_coerce_float := lambda value, default: _float_value(value, default), _FLOAT_FLAG),
    "_row_float": (_row_float := lambda row, column, default: _float_value(row[column], default) if column in row.index else float(default), _ROW_FLOAT_FLAG),
    "_event_index_value": (_event_index_value := lambda value: _exact_integral_value(value), _EVENT_INDEX_FLAG),
    "_successful_finite_scores": (_successful_finite_scores := lambda group: _successful_scores(group), _SUCCESS_FLAG),
    "_comparable_mask": (_comparable_mask := lambda frame: _comparison_mask(frame), _COMPARABLE_FLAG),
}
_TRUE_FLOAT = {"true", "yes", "y"}
_FALSE_FLOAT = {"false", "no", "n"}
_MISSING_STATUS = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_BYTES = (bytes, bytearray, memoryview)
_MISSING_KEY = object()
_NONSCALAR = object()
_SOURCE_COLUMNS = ("source_recovery_score_file",)
_INVALID_UTF8_KEY_PREFIX = "<invalid-utf8-event-key:"


def apply_recovery_diagnostics_bool_patch() -> None:
    """Install robust recovery-diagnostic scalar and event-scope helpers."""
    from . import recovery_diagnostics as diagnostics
    from . import simulation_recovery as recovery

    current = getattr(diagnostics, _PATCHED_FLAG, False) and all(getattr(getattr(diagnostics, name, None), flag, False) for name, (_, flag) in _HELPERS.items())
    current &= getattr(getattr(recovery, "_coerce_bool_series", None), _BOOL_SERIES_FLAG, False)
    current &= getattr(getattr(recovery, "_comparable_mask", None), _COMPARABLE_FLAG, False)
    current &= getattr(getattr(diagnostics, "_event_diagnostics", None), _EVENT_DIAGNOSTICS_FLAG, False)
    current &= _certified_current(diagnostics)
    if current:
        return

    for name, (helper, flag) in _HELPERS.items():
        setattr(helper, flag, True)
        setattr(diagnostics, name, helper)
    recovery._coerce_bool_series = _bool_series
    recovery._comparable_mask = _comparable_mask
    _install_certified_wrappers(diagnostics, recovery)
    _install_event_diagnostics(diagnostics)
    setattr(diagnostics, _PATCHED_FLAG, True)


def _unwrap(value: object) -> object:
    seen: set[int] = set()
    while isinstance(value, (np.ndarray, list, tuple, np.generic)):
        if isinstance(value, np.ndarray):
            if value.size != 1 or id(value) in seen:
                return _NONSCALAR
            seen.add(id(value))
            value = value.reshape(-1)[0]
        elif isinstance(value, (list, tuple)):
            if len(value) != 1 or id(value) in seen:
                return _NONSCALAR
            seen.add(id(value))
            value = value[0]
        else:
            value = value.item()
    return value


def _decode(value: object) -> object:
    value = _unwrap(value)
    if value is _NONSCALAR or not isinstance(value, _BYTES):
        return value
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError:
        return _NONSCALAR


def _is_missing(value: object) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _bool_value(value: object, default: bool = False) -> bool:
    value = _decode(value)
    if value is _NONSCALAR:
        return bool(default)
    return bool(_coerce_bool_series(pd.Series([value]), default=bool(default)).iloc[0])


def _map_bool(values: object, default: bool = False) -> pd.Series:
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    return series.map(lambda value: _bool_value(value, default)).astype(bool)


def _float_value(value: object, default: float) -> float:
    value = _decode(value)
    if value is _NONSCALAR or _is_missing(value):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        text = str(value).strip().lower()
        if text in _TRUE_FLOAT:
            return 1.0
        if text in _FALSE_FLOAT:
            return 0.0
        return float(default)


def _text(value: object) -> str | None:
    value = _decode(value)
    return None if value is _NONSCALAR or _is_missing(value) else str(value).strip()


def _status_ok(value: object) -> bool:
    value = _decode(value)
    if value is _NONSCALAR:
        return False
    if _is_missing(value):
        return True
    text = str(value).strip().lower()
    return text == "success" or text in _MISSING_STATUS


def _successful_scores(group: pd.DataFrame) -> pd.DataFrame:
    status = group["status"].map(_status_ok).astype(bool) if "status" in group else pd.Series(True, index=group.index)
    if "log_evidence" not in group:
        numeric = np.zeros(len(group), dtype=float)
    else:
        try:
            numeric = pd.to_numeric(group["log_evidence"], errors="coerce").to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError):
            numeric = group["log_evidence"].map(lambda value: _float_value(value, np.nan)).to_numpy(dtype=float)
    return group[status & pd.Series(np.isfinite(numeric), index=group.index)].copy()


def _comparison_mask(frame: pd.DataFrame) -> pd.Series:
    if "evidence_comparable" in frame:
        return _map_bool(frame["evidence_comparable"])
    if "evidence_support" in frame:
        return frame["evidence_support"].map(_text).fillna("").eq("exact_full_grid")
    return pd.Series(True, index=frame.index)


def _exact_integral_value(value: object) -> object:
    decoded = _decode(value)
    if decoded is _NONSCALAR:
        normalized = _normalize_key(value)
        return np.nan if normalized is _MISSING_KEY else normalized
    value = decoded
    if isinstance(value, str) and value.strip().lower() in _MISSING_STATUS:
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    try:
        return int(exact_index(value))
    except (TypeError, ValueError, OverflowError):
        pass
    if isinstance(value, str):
        try:
            numeric = Decimal(value.strip())
        except InvalidOperation:
            return value
        return int(numeric) if numeric.is_finite() and numeric == numeric.to_integral_value() else value
    if isinstance(value, Decimal):
        return int(value) if value.is_finite() and value == value.to_integral_value() else value
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return int(numeric) if np.isfinite(numeric) and numeric.is_integer() else value
    try:
        integer = int(value)
        return integer if value == integer else value
    except (TypeError, ValueError, OverflowError):
        return value


def _certified_current(diagnostics: Any) -> bool:
    event_current = getattr(
        getattr(diagnostics, "certified_vs_exact_event_recovery", None),
        _CERTIFIED_EVENT_FLAG,
        False,
    )
    summary_current = getattr(
        getattr(diagnostics, "certified_vs_exact_recovery_summary", None),
        _CERTIFIED_SUMMARY_FLAG,
        False,
    )
    return bool(event_current and summary_current)


def _install_certified_wrappers(diagnostics: Any, recovery: Any) -> None:
    if _certified_current(diagnostics):
        return

    def events(scores: pd.DataFrame) -> pd.DataFrame:
        if scores.empty:
            return recovery.certified_vs_exact_event_recovery(scores)
        pieces = []
        for _, group in _groups(scores):
            normalized = group.copy()
            if "status" in normalized:
                normalized["status"] = normalized["status"].map(lambda value: "success" if _status_ok(value) else value)
            piece = recovery.certified_vs_exact_event_recovery(normalized)
            if piece.empty:
                continue
            for column in _group_columns(group):
                if column not in piece:
                    piece[column] = _event_scalar(group, column)
            pieces.append(piece)
        return pd.DataFrame() if not pieces else _sort_events(pd.concat(pieces, ignore_index=True, sort=False))

    def summary(scores: pd.DataFrame) -> pd.DataFrame:
        event_rows = events(scores)
        if event_rows.empty:
            return pd.DataFrame()
        rows = [_summary_row(str(label), group) for label, group in event_rows.groupby("true_model", sort=False)]
        rows.append(_summary_row("overall", event_rows))
        return pd.DataFrame(rows)

    setattr(events, _CERTIFIED_EVENT_FLAG, True)
    setattr(summary, _CERTIFIED_SUMMARY_FLAG, True)
    diagnostics.certified_vs_exact_event_recovery = events
    diagnostics.certified_vs_exact_recovery_summary = summary


def _summary_row(label: str, group: pd.DataFrame) -> dict[str, object]:
    recovered = _map_bool(group["certified_vs_exact_recovered_expected_model"])
    margins = pd.to_numeric(group["expected_minus_best_comparable_log_evidence"], errors="coerce")
    n = len(group)
    return {
        "true_model": label,
        "expected_model": "" if label == "overall" else str(group["expected_model"].iloc[0]),
        "simulated_events": n,
        "certified_vs_exact_recovered_events": int(recovered.sum()),
        "certified_vs_exact_recovery_accuracy": _fraction(int(recovered.sum()), n),
        "mean_expected_minus_best_comparable_log_evidence": float(margins.mean()),
        "median_expected_minus_best_comparable_log_evidence": float(margins.median()),
        "events_without_comparable_exact_reference": int((group["certified_vs_exact_reason"] == "no_comparable_exact_reference").sum()),
    }


def _install_event_diagnostics(diagnostics: Any) -> None:
    if getattr(getattr(diagnostics, "_event_diagnostics", None), _EVENT_DIAGNOSTICS_FLAG, False):
        return

    def event_diagnostics(scores: pd.DataFrame, certified: pd.DataFrame) -> pd.DataFrame:
        columns = _group_columns(scores)
        lookup_columns = [column for column in columns if column in certified]
        lookup = {_row_key(row, lookup_columns): row for _, row in certified.iterrows()}
        rows = []
        for _, group in _groups(scores):
            first = group.iloc[0]
            row = diagnostics._event_diagnostic_row(str(first.get("session", "")), first.get("event_index", np.nan), group, lookup.get(_row_key(first, lookup_columns)))
            for column in columns:
                row.setdefault(column, _event_scalar(group, column))
            rows.append(row)
        return _sort_events(pd.DataFrame(rows))

    setattr(event_diagnostics, _EVENT_DIAGNOSTICS_FLAG, True)
    diagnostics._event_diagnostics = event_diagnostics


def _group_columns(frame: pd.DataFrame) -> list[str]:
    from . import simulation_best_row_flags

    columns = list(simulation_best_row_flags._event_group_columns(frame))
    for column in reversed(_SOURCE_COLUMNS):
        if column in frame and column not in columns:
            columns.insert(0, column)
    return columns or [column for column in ("session", "event_index") if column in frame]


def _groups(frame: pd.DataFrame) -> Any:
    normalized = _normalized_group_frame(frame)
    columns = _group_columns(normalized)
    return normalized.groupby(columns, sort=False, dropna=False) if columns else [((), normalized)]


def _normalized_group_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a score table with hashable semantic event-identity values."""
    normalized = frame.copy()
    for column in _group_columns(normalized):
        if column in normalized:
            normalized[column] = normalized[column].map(_group_value)
    return normalized


def _group_value(value: object) -> object:
    normalized = _normalize_key(value)
    return np.nan if normalized is _MISSING_KEY else normalized


def _nested_key(value: object, seen: set[int]) -> object:
    normalized = _normalize_key(value, seen)
    return ("missing",) if normalized is _MISSING_KEY else normalized


def _normalize_key(value: object, seen: set[int] | None = None) -> object:
    if seen is None:
        seen = set()
    if isinstance(value, np.generic):
        return _normalize_key(value.item(), seen)
    if isinstance(value, np.ndarray):
        container_id = id(value)
        if container_id in seen:
            return ("object", "<cyclic-array>")
        if value.size == 1:
            seen.add(container_id)
            try:
                return _normalize_key(value.reshape(-1)[0], seen)
            finally:
                seen.remove(container_id)
        seen.add(container_id)
        try:
            items = tuple(_nested_key(item, seen) for item in value.reshape(-1))
        finally:
            seen.remove(container_id)
        return ("sequence", items) if value.ndim == 1 else ("array", tuple(value.shape), items)
    if isinstance(value, _BYTES):
        raw = bytes(value)
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"{_INVALID_UTF8_KEY_PREFIX}{raw.hex()}>"
    if isinstance(value, str):
        text = value.strip()
        return _MISSING_KEY if text.lower() in _MISSING_STATUS else text
    if isinstance(value, (list, tuple)):
        container_id = id(value)
        if container_id in seen:
            return ("object", "<cyclic-sequence>")
        seen.add(container_id)
        try:
            if len(value) == 1:
                return _normalize_key(value[0], seen)
            return ("sequence", tuple(_nested_key(item, seen) for item in value))
        finally:
            seen.remove(container_id)
    if isinstance(value, (set, frozenset)):
        container_id = id(value)
        if container_id in seen:
            return ("object", "<cyclic-set>")
        seen.add(container_id)
        try:
            items = (_nested_key(item, seen) for item in value)
            return ("set", tuple(sorted(items, key=repr)))
        finally:
            seen.remove(container_id)
    if isinstance(value, dict):
        container_id = id(value)
        if container_id in seen:
            return ("object", "<cyclic-mapping>")
        seen.add(container_id)
        try:
            items = (
                (_nested_key(key, seen), _nested_key(item, seen))
                for key, item in value.items()
            )
            return ("mapping", tuple(sorted(items, key=repr)))
        finally:
            seen.remove(container_id)
    if _is_missing(value):
        return _MISSING_KEY
    try:
        hash(value)
    except TypeError:
        return ("object", repr(value))
    return value


def _row_key(row: pd.Series, columns: list[str]) -> tuple[object, ...]:
    return tuple(_normalize_key(row.get(column, np.nan)) for column in columns)


def _sort_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    columns = [column for column in _group_columns(frame) if column in frame]
    if not columns:
        return frame.reset_index(drop=True)
    keys = pd.DataFrame(
        {
            column: frame[column].map(
                lambda value: "" if _normalize_key(value) is _MISSING_KEY else str(_normalize_key(value))
            )
            for column in columns
        },
        index=frame.index,
    )
    return frame.loc[keys.sort_values(columns, kind="mergesort").index].reset_index(drop=True)


def _event_scalar(group: pd.DataFrame, column: str) -> object:
    return group[column].iloc[0] if column in group and not group.empty else np.nan


def _fraction(numerator: int, denominator: int) -> float:
    return np.nan if denominator <= 0 else numerator / denominator


__all__ = ["apply_recovery_diagnostics_bool_patch"]
