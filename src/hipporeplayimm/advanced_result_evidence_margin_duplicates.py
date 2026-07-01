"""Keep evidence-margin diagnostics from comparing duplicate rows of one model."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_evidence_margin_distinct_model_patch_applied"
_MARGIN_FLAG = "_evidence_margin_distinct_model_wrapper"
_ADD_FLAG = "_evidence_margin_distinct_model_add_columns_wrapper"
_ORIGINAL_MARGIN = "_evidence_margin_distinct_model_original"
_ORIGINAL_ADD = "_evidence_margin_distinct_model_original_add"
_DEFAULT_GROUP_COLUMNS = ("session", "event_index")


def apply_evidence_margin_distinct_model_patch() -> None:
    """Install distinct-model evidence margin diagnostics."""

    from . import advanced_result_diagnostics as diagnostics

    current_margin = diagnostics.evidence_margin_table
    current_add = diagnostics.add_evidence_margin_columns
    if getattr(diagnostics, _PATCHED_FLAG, False) and getattr(current_margin, _MARGIN_FLAG, False) and getattr(current_add, _ADD_FLAG, False):
        return

    original_margin = getattr(current_margin, _ORIGINAL_MARGIN, current_margin)

    @wraps(original_margin)
    def evidence_margin_table(
        scores: pd.DataFrame,
        *,
        group_cols: Sequence[str] = _DEFAULT_GROUP_COLUMNS,
        evidence_col: str = "log_evidence",
        model_col: str = "model",
    ) -> pd.DataFrame:
        groups = _normalize_group_cols(group_cols)
        collapsed = _collapse_duplicate_models(diagnostics, scores, groups, evidence_col, model_col)
        if collapsed is None:
            return original_margin(scores, group_cols=groups, evidence_col=evidence_col, model_col=model_col)
        return original_margin(collapsed, group_cols=groups, evidence_col=evidence_col, model_col=model_col)

    original_add = getattr(current_add, _ORIGINAL_ADD, current_add)

    @wraps(original_add)
    def add_evidence_margin_columns(
        scores: pd.DataFrame,
        *,
        group_cols: Sequence[str] = _DEFAULT_GROUP_COLUMNS,
    ) -> pd.DataFrame:
        groups = _normalize_group_cols(group_cols)
        if scores.empty:
            return scores.copy()
        margins = diagnostics.evidence_margin_table(scores, group_cols=groups)
        if margins.empty:
            out = scores.copy()
            out["evidence_margin_to_second_best"] = np.nan
            out["evidence_margin_category"] = "missing"
            return out
        margins = margins.copy()
        for column in groups:
            if column in scores.columns and column in margins.columns:
                try:
                    margins[column] = margins[column].astype(scores[column].dtype)
                except (TypeError, ValueError):
                    pass
        return scores.merge(margins, on=list(groups), how="left")

    setattr(evidence_margin_table, _MARGIN_FLAG, True)
    setattr(evidence_margin_table, _ORIGINAL_MARGIN, original_margin)
    setattr(add_evidence_margin_columns, _ADD_FLAG, True)
    setattr(add_evidence_margin_columns, _ORIGINAL_ADD, original_add)
    diagnostics.evidence_margin_table = evidence_margin_table
    diagnostics.add_evidence_margin_columns = add_evidence_margin_columns
    setattr(diagnostics, _PATCHED_FLAG, True)


def _collapse_duplicate_models(diagnostics: object, scores: pd.DataFrame, group_cols: tuple[str, ...], evidence_col: str, model_col: str) -> pd.DataFrame | None:
    if scores.empty:
        return None
    comparable = diagnostics._comparable_rows(scores)
    if comparable.empty:
        return comparable
    missing = [column for column in (*group_cols, evidence_col, model_col) if column not in comparable.columns]
    if missing:
        return None
    rows: list[pd.DataFrame] = []
    grouped = comparable.groupby(list(group_cols), sort=False, dropna=False) if group_cols else (((), comparable),)
    for _, group in grouped:
        numeric = group.copy()
        numeric[evidence_col] = pd.to_numeric(numeric[evidence_col], errors="coerce")
        numeric = numeric.dropna(subset=[evidence_col]).sort_values(evidence_col, ascending=False, kind="stable")
        if not numeric.empty:
            rows.append(numeric.drop_duplicates(model_col, keep="first"))
    if not rows:
        return comparable.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=False)


def _normalize_group_cols(group_cols: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(group_cols, str):
        return (group_cols,)
    return tuple(group_cols)


__all__ = ["apply_evidence_margin_distinct_model_patch"]
