"""Guard simulation-recovery best-row selection against stale flags."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


_GROUP_COLUMNS = (
    "session",
    "event_index",
    "window_index",
    "benchmark_cell_split_index",
    "event_window_variant",
)
_PATCHED_FLAG = "_simulation_best_row_flag_scope_patch_applied"
_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}


def apply_simulation_best_row_flags_patch() -> None:
    """Install per-event guarded handling of explicit best-model flags."""

    from . import evidence_reporting as reporting
    from . import simulation_recovery as recovery

    current = reporting.simulation_event_best_rows
    if getattr(current, _PATCHED_FLAG, False):
        return

    @wraps(current)
    def simulation_event_best_rows_with_scoped_flags(event_scores: pd.DataFrame) -> pd.DataFrame:
        scored = reporting.ensure_evidence_support_columns(event_scores)
        if scored.empty:
            return _empty_like(scored)
        comparable = reporting._coerce_bool_series(scored["evidence_comparable"])
        status_ok = _status_success_mask(scored)
        ok = scored[status_ok & comparable]
        if ok.empty:
            return _empty_like(ok)
        ok = _finite_log_evidence_rows(ok)
        if ok.empty:
            return _empty_like(ok)
        if "is_best_model" not in ok.columns:
            return _best_by_log_evidence(ok)
        return _best_rows_with_guarded_flags(ok, reporting)

    setattr(simulation_event_best_rows_with_scoped_flags, _PATCHED_FLAG, True)
    reporting.simulation_event_best_rows = simulation_event_best_rows_with_scoped_flags
    recovery._event_best_rows = simulation_event_best_rows_with_scoped_flags


def _status_success_mask(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["status"].map(_status_is_success).astype(bool)


def _status_is_success(value: object) -> bool:
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    return text == "success" or text in _MISSING_STATUS_VALUES


def _best_rows_with_guarded_flags(frame: pd.DataFrame, reporting: Any) -> pd.DataFrame:
    group_columns = _event_group_columns(frame)
    if not group_columns:
        flags = reporting._coerce_bool_series(frame["is_best_model"])
        if int(flags.sum()) == 1:
            return frame.loc[flags].reset_index(drop=True)
        return _best_by_log_evidence(frame)

    pieces = []
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        flags = reporting._coerce_bool_series(group["is_best_model"])
        if int(flags.sum()) == 1:
            pieces.append(group.loc[flags])
        else:
            pieces.append(_best_by_log_evidence(group))
    if not pieces:
        return _empty_like(frame)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _finite_log_evidence_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_like(frame)
    working = frame.copy()
    working["log_evidence"] = pd.to_numeric(working["log_evidence"], errors="coerce")
    return working[np.isfinite(working["log_evidence"].to_numpy(dtype=float))].copy()


def _best_by_log_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_like(frame)
    working = _finite_log_evidence_rows(frame)
    if working.empty:
        return _empty_like(working)
    group_columns = _event_group_columns(working)
    sort_columns = [*group_columns, "log_evidence"]
    ascending = [True] * len(group_columns) + [False]
    best = working.sort_values(sort_columns, ascending=ascending)
    if group_columns:
        best = best.drop_duplicates(group_columns, keep="first")
    else:
        best = best.head(1)
    return best.reset_index(drop=True)


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[0:0].copy()


def _event_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in _GROUP_COLUMNS if column in frame.columns]


__all__ = ["apply_simulation_best_row_flags_patch"]
