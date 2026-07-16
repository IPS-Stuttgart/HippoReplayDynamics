"""Keep evidence-margin diagnostics finite and distinct by model."""

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_evidence_margin_distinct_model_patch_applied"
_MARGIN_FLAG = "_evidence_margin_distinct_model_wrapper"
_ADD_FLAG = "_evidence_margin_distinct_model_add_columns_wrapper"
_BOOL_FLAG = "_evidence_margin_arbitrary_integer_bool_wrapper"
_DEFAULT_GROUP_COLUMNS = ("session", "event_index")
_MARGIN_COLUMNS = (
    "best_model_by_evidence",
    "second_best_model_by_evidence",
    "best_log_evidence",
    "second_best_log_evidence",
    "evidence_margin_to_second_best",
    "evidence_margin_category",
    "models_compared",
)


def apply_evidence_margin_distinct_model_patch() -> None:
    """Install distinct-model evidence margin diagnostics."""

    from . import advanced_result_diagnostics as diagnostics

    current_margin = diagnostics.evidence_margin_table
    current_add = diagnostics.add_evidence_margin_columns
    current_bool = diagnostics._as_bool
    if (
        getattr(diagnostics, _PATCHED_FLAG, False)
        and getattr(current_margin, _MARGIN_FLAG, False)
        and getattr(current_add, _ADD_FLAG, False)
        and getattr(current_bool, _BOOL_FLAG, False)
    ):
        return
    original_margin = current_margin
    original_bool = current_bool

    def evidence_margin_table(scores, *, group_cols=_DEFAULT_GROUP_COLUMNS, evidence_col="log_evidence", model_col="model"):
        groups = _normalize_group_cols(group_cols)
        collapsed = _collapse_duplicate_models(diagnostics, scores, groups, evidence_col, model_col)
        if collapsed is None:
            return original_margin(scores, group_cols=groups, evidence_col=evidence_col, model_col=model_col)
        return original_margin(collapsed, group_cols=groups, evidence_col=evidence_col, model_col=model_col)

    def add_evidence_margin_columns(scores, *, group_cols=_DEFAULT_GROUP_COLUMNS):
        groups = _normalize_group_cols(group_cols)
        if scores.empty:
            return scores.copy()
        base = scores.drop(
            columns=[column for column in _MARGIN_COLUMNS if column in scores.columns]
        ).copy()
        margins = evidence_margin_table(base, group_cols=groups)
        if margins.empty:
            base["evidence_margin_to_second_best"] = np.nan
            base["evidence_margin_category"] = "missing"
            return base
        margins = _align_margin_group_key_dtypes(base, margins, groups)
        return _merge_margin_columns_preserving_index(base, margins, groups)

    @wraps(original_bool)
    def _as_bool(value: object, *, default: bool = False) -> bool:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(value) != 0
        return original_bool(value, default=default)

    setattr(evidence_margin_table, _MARGIN_FLAG, True)
    setattr(add_evidence_margin_columns, _ADD_FLAG, True)
    setattr(_as_bool, _BOOL_FLAG, True)
    diagnostics.evidence_margin_table = evidence_margin_table
    diagnostics.add_evidence_margin_columns = add_evidence_margin_columns
    diagnostics._as_bool = _as_bool
    setattr(diagnostics, _PATCHED_FLAG, True)


def _collapse_duplicate_models(diagnostics, scores: pd.DataFrame, group_cols: tuple[str, ...], evidence_col: str, model_col: str) -> pd.DataFrame | None:
    if scores.empty:
        return None
    ok = diagnostics._comparable_rows(scores)
    if ok.empty:
        return ok
    if any(column not in ok.columns for column in (*group_cols, evidence_col, model_col)):
        return None
    rows = []
    grouped = ok.groupby(list(group_cols), sort=False, dropna=False) if group_cols else (((), ok),)
    for _, group in grouped:
        group = group.copy()
        numeric_evidence = pd.to_numeric(group[evidence_col], errors="coerce")
        finite_evidence = numeric_evidence.notna() & np.isfinite(numeric_evidence)
        group = group.loc[finite_evidence].copy()
        group[evidence_col] = numeric_evidence.loc[finite_evidence].astype(float)
        group = group.sort_values(evidence_col, ascending=False, kind="stable")
        if not group.empty:
            rows.append(group.drop_duplicates(model_col, keep="first"))
    return pd.concat(rows, ignore_index=False) if rows else ok.iloc[0:0].copy()


def _align_margin_group_key_dtypes(
    scores: pd.DataFrame,
    margins: pd.DataFrame,
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Match margin grouping-key dtypes before merging back into score rows."""

    out = margins.copy()
    for column in group_cols:
        if column in scores.columns and column in out.columns:
            out[column] = out[column].astype(scores[column].dtype)
    return out


def _merge_margin_columns_preserving_index(
    scores: pd.DataFrame,
    margins: pd.DataFrame,
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Attach one margin row per group without replacing the caller's index."""

    row_position_column = "__hipporeplayimm_margin_row_position__"
    while row_position_column in scores.columns or row_position_column in margins.columns:
        row_position_column += "_"

    original_index = scores.index.copy()
    left = scores.reset_index(drop=True).copy()
    left[row_position_column] = np.arange(len(left), dtype=np.int64)
    merged = left.merge(
        margins,
        on=list(group_cols),
        how="left",
        sort=False,
        validate="many_to_one",
    )
    merged = merged.sort_values(row_position_column, kind="stable").drop(
        columns=[row_position_column]
    )
    merged.index = original_index
    return merged


def _normalize_group_cols(group_cols):
    return (group_cols,) if isinstance(group_cols, str) else tuple(group_cols)


__all__ = ["apply_evidence_margin_distinct_model_patch"]
