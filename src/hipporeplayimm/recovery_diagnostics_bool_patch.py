"""Use patched scalar parsing and event scoping in recovery diagnostics.

Recovery diagnostic tables are often rebuilt from CSV score artifacts.  Pandas can
round-trip boolean columns as strings such as ``"1.0"`` and ``"0.0"``.  The
shared evidence-reporting parser already handles those values; this patch makes
the diagnostic scalar helpers use the same semantics.

The diagnostic event table must also use the same event identity as the recovery
summary helpers.  Concatenated recovery artifacts can reuse ``session`` and
``event_index`` across source files, random seeds, or rescoring windows, so
diagnostics must not collapse those rows back into one event.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from operator import index as exact_index
from typing import Any

import numpy as np
import pandas as pd

from .evidence_reporting import _coerce_bool_series

_PATCHED_FLAG = "_recovery_diagnostics_bool_scalar_patch_applied"
_COERCE_BOOL_WRAPPER_FLAG = "_recovery_diagnostics_bool_coerce_bool_wrapper"
_ROW_BOOL_WRAPPER_FLAG = "_recovery_diagnostics_bool_row_bool_wrapper"
_COERCE_FLOAT_WRAPPER_FLAG = "_recovery_diagnostics_bool_coerce_float_wrapper"
_ROW_FLOAT_WRAPPER_FLAG = "_recovery_diagnostics_bool_row_float_wrapper"
_EVENT_INDEX_VALUE_WRAPPER_FLAG = "_recovery_diagnostics_exact_event_index_wrapper"
_SUCCESSFUL_FINITE_SCORES_WRAPPER_FLAG = "_recovery_diagnostics_bool_successful_finite_scores_wrapper"
_EVENT_DIAGNOSTICS_WRAPPER_FLAG = "_recovery_diagnostics_scoped_event_diagnostics_wrapper"
_CERTIFIED_EVENT_WRAPPER_FLAG = "_recovery_diagnostics_scoped_certified_event_wrapper"
_CERTIFIED_SUMMARY_WRAPPER_FLAG = "_recovery_diagnostics_scoped_certified_summary_wrapper"
_HELPER_FLAGS = {
    "_coerce_bool": _COERCE_BOOL_WRAPPER_FLAG,
    "_row_bool": _ROW_BOOL_WRAPPER_FLAG,
    "_coerce_float": _COERCE_FLOAT_WRAPPER_FLAG,
    "_row_float": _ROW_FLOAT_WRAPPER_FLAG,
    "_event_index_value": _EVENT_INDEX_VALUE_WRAPPER_FLAG,
    "_successful_finite_scores": _SUCCESSFUL_FINITE_SCORES_WRAPPER_FLAG,
}
_TRUE_FLOAT_STRINGS = {"true", "yes", "y"}
_FALSE_FLOAT_STRINGS = {"false", "no", "n"}
_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_MISSING_KEY_VALUE = object()
_SOURCE_GROUP_COLUMNS = ("source_recovery_score_file",)


def apply_recovery_diagnostics_bool_patch() -> None:
    """Install shared scalar bool coercion and scoped event diagnostics."""

    from . import recovery_diagnostics as diagnostics
    from . import simulation_recovery as recovery

    if (
        getattr(diagnostics, _PATCHED_FLAG, False)
        and _helpers_are_patched(diagnostics)
        and _event_diagnostics_is_patched(diagnostics)
        and _certified_wrappers_are_patched(diagnostics)
    ):
        return

    def coerce_bool(value: object, default: bool = False) -> bool:
        try:
            if pd.isna(value):
                return bool(default)
        except (TypeError, ValueError):
            return bool(default)
        return bool(_coerce_bool_series(pd.Series([value]), default=bool(default)).iloc[0])

    def row_bool(row: Any, column: str, default: bool) -> bool:
        if column not in row.index:
            return bool(default)
        return coerce_bool(row[column], default)

    def coerce_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            text = str(value).strip().lower()
            if text in _TRUE_FLOAT_STRINGS:
                return 1.0
            if text in _FALSE_FLOAT_STRINGS:
                return 0.0
            return float(default)

    def row_float(row: Any, column: str, default: float) -> float:
        if column not in row.index:
            return float(default)
        try:
            if pd.isna(row[column]):
                return float(default)
        except (TypeError, ValueError):
            return float(default)
        return coerce_float(row[column], default)

    def event_index_value(value: object) -> object:
        return _exact_integral_value(value)

    def successful_finite_scores(group: pd.DataFrame) -> pd.DataFrame:
        if "status" in group:
            status_ok = group["status"].map(_status_is_success_or_missing).astype(bool)
        else:
            status_ok = pd.Series(True, index=group.index)
        if "log_evidence" in group:
            try:
                values = pd.to_numeric(group["log_evidence"], errors="coerce")
                numeric_values = values.to_numpy(dtype=float)
            except (TypeError, ValueError, OverflowError):
                numeric_values = (
                    group["log_evidence"]
                    .map(lambda value: coerce_float(value, float("nan")))
                    .to_numpy(dtype=float)
                )
        else:
            numeric_values = np.zeros(len(group), dtype=float)
        finite = pd.Series(np.isfinite(numeric_values), index=group.index)
        return group[status_ok & finite].copy()

    _install_helper(diagnostics, "_coerce_bool", coerce_bool)
    _install_helper(diagnostics, "_row_bool", row_bool)
    _install_helper(diagnostics, "_coerce_float", coerce_float)
    _install_helper(diagnostics, "_row_float", row_float)
    _install_helper(diagnostics, "_event_index_value", event_index_value)
    _install_helper(diagnostics, "_successful_finite_scores", successful_finite_scores)
    _install_scoped_certified_recovery(diagnostics, recovery)
    _install_scoped_event_diagnostics(diagnostics)
    setattr(diagnostics, _PATCHED_FLAG, True)


def _exact_integral_value(value: object) -> object:
    """Normalize integral event IDs without a lossy binary-float round trip."""

    if isinstance(value, (bool, np.bool_)):
        return int(value)
    try:
        return int(exact_index(value))
    except (TypeError, ValueError, OverflowError):
        pass

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            decimal_value = Decimal(text)
        except InvalidOperation:
            return value
        if decimal_value.is_finite() and decimal_value == decimal_value.to_integral_value():
            return int(decimal_value)
        return value

    if isinstance(value, Decimal):
        if value.is_finite() and value == value.to_integral_value():
            return int(value)
        return value

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        return value

    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return value
    try:
        return integer if value == integer else value
    except (TypeError, ValueError):
        return value


def _helpers_are_patched(diagnostics: Any) -> bool:
    """Return whether all recovery-diagnostic scalar helpers are active wrappers."""

    return all(
        getattr(getattr(diagnostics, helper_name, None), wrapper_flag, False)
        for helper_name, wrapper_flag in _HELPER_FLAGS.items()
    )


def _event_diagnostics_is_patched(diagnostics: Any) -> bool:
    return bool(getattr(getattr(diagnostics, "_event_diagnostics", None), _EVENT_DIAGNOSTICS_WRAPPER_FLAG, False))


def _certified_wrappers_are_patched(diagnostics: Any) -> bool:
    return bool(
        getattr(getattr(diagnostics, "certified_vs_exact_event_recovery", None), _CERTIFIED_EVENT_WRAPPER_FLAG, False)
        and getattr(getattr(diagnostics, "certified_vs_exact_recovery_summary", None), _CERTIFIED_SUMMARY_WRAPPER_FLAG, False)
    )


def _install_helper(diagnostics: Any, helper_name: str, helper: Any) -> None:
    setattr(helper, _HELPER_FLAGS[helper_name], True)
    setattr(diagnostics, helper_name, helper)


def _install_scoped_certified_recovery(diagnostics: Any, recovery: Any) -> None:
    if _certified_wrappers_are_patched(diagnostics):
        return

    def certified_vs_exact_event_recovery(event_scores: pd.DataFrame) -> pd.DataFrame:
        return _certified_vs_exact_event_recovery_with_scoped_keys(event_scores, recovery)

    def certified_vs_exact_recovery_summary(event_scores: pd.DataFrame) -> pd.DataFrame:
        events = certified_vs_exact_event_recovery(event_scores)
        return _certified_vs_exact_summary_from_events(events)

    setattr(certified_vs_exact_event_recovery, _CERTIFIED_EVENT_WRAPPER_FLAG, True)
    setattr(certified_vs_exact_recovery_summary, _CERTIFIED_SUMMARY_WRAPPER_FLAG, True)
    diagnostics.certified_vs_exact_event_recovery = certified_vs_exact_event_recovery
    diagnostics.certified_vs_exact_recovery_summary = certified_vs_exact_recovery_summary


def _install_scoped_event_diagnostics(diagnostics: Any) -> None:
    if _event_diagnostics_is_patched(diagnostics):
        return

    def event_diagnostics(scores: pd.DataFrame, certified_events: pd.DataFrame) -> pd.DataFrame:
        return _event_diagnostics_with_scoped_keys(scores, certified_events, diagnostics)

    setattr(event_diagnostics, _EVENT_DIAGNOSTICS_WRAPPER_FLAG, True)
    diagnostics._event_diagnostics = event_diagnostics


def _certified_vs_exact_event_recovery_with_scoped_keys(
    event_scores: pd.DataFrame,
    recovery: Any,
) -> pd.DataFrame:
    if event_scores.empty:
        return recovery.certified_vs_exact_event_recovery(event_scores)

    pieces: list[pd.DataFrame] = []
    for _, group in _iter_event_groups(event_scores):
        if group.empty:
            continue
        piece = recovery.certified_vs_exact_event_recovery(group)
        if piece.empty:
            continue
        group_columns = _event_group_columns(group)
        piece = piece.copy()
        for column in group_columns:
            if column not in piece.columns:
                piece[column] = _event_scalar(group, column)
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame()
    return _sort_by_event_group_columns(pd.concat(pieces, ignore_index=True, sort=False))


def _certified_vs_exact_summary_from_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = [_certified_vs_exact_summary_row(str(true_model), group) for true_model, group in events.groupby("true_model", sort=False)]
    rows.append(_certified_vs_exact_summary_row("overall", events))
    return pd.DataFrame(rows)


def _certified_vs_exact_summary_row(label: str, group: pd.DataFrame) -> dict[str, object]:
    recovered = _coerce_bool_series(group["certified_vs_exact_recovered_expected_model"])
    margins = pd.to_numeric(group["expected_minus_best_comparable_log_evidence"], errors="coerce")
    expected_model = "" if label == "overall" else str(group["expected_model"].iloc[0])
    n_events = int(len(group))
    return {
        "true_model": label,
        "expected_model": expected_model,
        "simulated_events": n_events,
        "certified_vs_exact_recovered_events": int(recovered.sum()),
        "certified_vs_exact_recovery_accuracy": _safe_fraction(int(recovered.sum()), n_events),
        "mean_expected_minus_best_comparable_log_evidence": float(margins.mean()),
        "median_expected_minus_best_comparable_log_evidence": float(margins.median()),
        "events_without_comparable_exact_reference": int((group["certified_vs_exact_reason"] == "no_comparable_exact_reference").sum()),
    }


def _event_diagnostics_with_scoped_keys(
    scores: pd.DataFrame,
    certified_events: pd.DataFrame,
    diagnostics: Any,
) -> pd.DataFrame:
    group_columns = _event_group_columns(scores)
    lookup_columns = [column for column in group_columns if column in certified_events.columns]
    certified_lookup = {
        _row_key(row, lookup_columns): row
        for _, row in certified_events.iterrows()
    }

    rows: list[dict[str, object]] = []
    for _, group in _iter_event_groups(scores):
        if group.empty:
            continue
        first = group.iloc[0]
        session = first.get("session", "")
        event_index = first.get("event_index", np.nan)
        row = diagnostics._event_diagnostic_row(
            str(session),
            event_index,
            group,
            certified_lookup.get(_row_key(first, lookup_columns)),
        )
        for column in group_columns:
            if column not in row:
                row[column] = _event_scalar(group, column)
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return _sort_by_event_group_columns(result)


def _iter_event_groups(frame: pd.DataFrame) -> Any:
    group_columns = _event_group_columns(frame)
    if group_columns:
        return frame.groupby(group_columns, sort=False, dropna=False)
    return [((), frame)]


def _event_group_columns(frame: pd.DataFrame) -> list[str]:
    from . import simulation_best_row_flags as best_row_flags

    group_columns = list(best_row_flags._event_group_columns(frame))
    for column in reversed(_SOURCE_GROUP_COLUMNS):
        if column in frame.columns and column not in group_columns:
            group_columns.insert(0, column)
    if group_columns:
        return group_columns
    return [column for column in ("session", "event_index") if column in frame.columns]


def _sort_by_event_group_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    sort_columns = [column for column in _event_group_columns(frame) if column in frame.columns]
    if not sort_columns:
        return frame.reset_index(drop=True)
    sort_keys = pd.DataFrame({column: frame[column].map(_sort_key_value) for column in sort_columns}, index=frame.index)
    order = sort_keys.sort_values(sort_columns, kind="mergesort").index
    return frame.loc[order].reset_index(drop=True)


def _sort_key_value(value: object) -> str:
    normalized = _normalize_key_value(value)
    if normalized is _MISSING_KEY_VALUE:
        return ""
    return str(normalized)


def _row_key(row: pd.Series, columns: list[str]) -> tuple[object, ...]:
    return tuple(_normalize_key_value(row.get(column, np.nan)) for column in columns)


def _normalize_key_value(value: object) -> object:
    try:
        if pd.isna(value):
            return _MISSING_KEY_VALUE
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _event_scalar(group: pd.DataFrame, column: str) -> object:
    return group[column].iloc[0] if column in group.columns and not group.empty else np.nan


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float("nan") if denominator <= 0 else float(numerator) / float(denominator)


def _status_is_success_or_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return True
    text = str(value).strip().lower()
    return text == "success" or text in _MISSING_STATUS_VALUES


__all__ = ["apply_recovery_diagnostics_bool_patch"]
