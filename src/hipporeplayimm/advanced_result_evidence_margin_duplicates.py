"""Keep evidence-margin diagnostics finite and distinct by model."""

from __future__ import annotations

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_evidence_margin_distinct_model_patch_applied"
_MARGIN_FLAG = "_evidence_margin_distinct_model_wrapper"
_ADD_FLAG = "_evidence_margin_distinct_model_add_columns_wrapper"
_DEFAULT_GROUP_COLUMNS = ("session", "event_index")


def apply_evidence_margin_distinct_model_patch() -> None:
    """Install distinct-model evidence margin diagnostics."""

    from . import advanced_result_diagnostics as diagnostics

    current_margin = diagnostics.evidence_margin_table
    current_add = diagnostics.add_evidence_margin_columns
    if getattr(diagnostics, _PATCHED_FLAG, False) and getattr(current_margin, _MARGIN_FLAG, False) and getattr(current_add, _ADD_FLAG, False):
        return
    original_margin = current_margin

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
        margins = evidence_margin_table(scores, group_cols=groups)
        if margins.empty:
            out = scores.copy()
            out["evidence_margin_to_second_best"] = np.nan
            out["evidence_margin_category"] = "missing"
            return out
        margins = _align_margin_group_key_dtypes(scores, margins, groups)
        return scores.merge(margins, on=list(groups), how="left")

    setattr(evidence_margin_table, _MARGIN_FLAG, True)
    setattr(add_evidence_margin_columns, _ADD_FLAG, True)
    diagnostics.evidence_margin_table = evidence_margin_table
    diagnostics.add_evidence_margin_columns = add_evidence_margin_columns
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


def _normalize_group_cols(group_cols):
    return (group_cols,) if isinstance(group_cols, str) else tuple(group_cols)


__all__ = ["apply_evidence_margin_distinct_model_patch"]
