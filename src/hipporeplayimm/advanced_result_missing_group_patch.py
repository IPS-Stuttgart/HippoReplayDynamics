"""Preserve advanced evidence-margin groups with missing scope metadata."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .wrong_map_missing_group_patch import apply_wrong_map_missing_group_patch, wrong_map_missing_group_patch_current

_PATCH_FLAG = "_missing_group_metadata_patch_applied"
_EVIDENCE_MARGIN_TABLE_WRAPPER_FLAG = "_missing_group_metadata_evidence_margin_table_wrapper"
_ADD_COLUMNS_WRAPPER_FLAG = "_missing_group_metadata_add_margin_columns_wrapper"
_MARGIN_COLUMNS = [
    "best_model_by_evidence",
    "second_best_model_by_evidence",
    "best_log_evidence",
    "second_best_log_evidence",
    "evidence_margin_to_second_best",
    "evidence_margin_category",
    "models_compared",
]


def apply_advanced_result_missing_group_patch() -> None:
    """Keep margin diagnostics for rows whose optional grouping metadata is missing."""

    from . import advanced_result_diagnostics as diagnostics

    if getattr(diagnostics, _PATCH_FLAG, False) and _missing_group_patch_current(diagnostics) and wrong_map_missing_group_patch_current(diagnostics):
        return

    def evidence_margin_table(
        scores: pd.DataFrame,
        *,
        group_cols: Sequence[str] = ("session", "event_index"),
        evidence_col: str = "log_evidence",
        model_col: str = "model",
    ) -> pd.DataFrame:
        """Return one calibrated evidence-margin row per event, retaining NA group keys."""

        ok = diagnostics._comparable_rows(scores)
        if ok.empty:
            return pd.DataFrame(columns=[*group_cols, *_MARGIN_COLUMNS])
        missing = [column for column in (*group_cols, evidence_col, model_col) if column not in ok.columns]
        if missing:
            raise KeyError(f"scores is missing required columns: {missing}")

        rows: list[dict[str, object]] = []
        for key, group in ok.groupby(list(group_cols), sort=False, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            group = group.dropna(subset=[evidence_col]).sort_values(evidence_col, ascending=False)
            if group.empty:
                continue
            best = group.iloc[0]
            second = group.iloc[1] if len(group) > 1 else None
            best_value = float(best[evidence_col])
            second_value = float(second[evidence_col]) if second is not None else np.nan
            margin = best_value - second_value if second is not None else np.inf
            row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
            row.update(
                {
                    "best_model_by_evidence": str(best[model_col]),
                    "second_best_model_by_evidence": "" if second is None else str(second[model_col]),
                    "best_log_evidence": best_value,
                    "second_best_log_evidence": second_value,
                    "evidence_margin_to_second_best": float(margin),
                    "evidence_margin_category": diagnostics.classify_evidence_margin(margin),
                    "models_compared": int(len(group)),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def add_evidence_margin_columns(
        scores: pd.DataFrame,
        *,
        group_cols: Sequence[str] = ("session", "event_index"),
    ) -> pd.DataFrame:
        """Merge event-level evidence-margin diagnostics back into score rows."""

        if scores.empty:
            return scores.copy()
        margins = evidence_margin_table(scores, group_cols=group_cols)
        if margins.empty:
            out = scores.copy()
            out["evidence_margin_to_second_best"] = np.nan
            out["evidence_margin_category"] = "missing"
            return out
        return scores.merge(margins, on=list(group_cols), how="left")

    setattr(evidence_margin_table, _EVIDENCE_MARGIN_TABLE_WRAPPER_FLAG, True)
    setattr(add_evidence_margin_columns, _ADD_COLUMNS_WRAPPER_FLAG, True)
    diagnostics.evidence_margin_table = evidence_margin_table
    diagnostics.add_evidence_margin_columns = add_evidence_margin_columns
    apply_wrong_map_missing_group_patch(diagnostics)
    setattr(diagnostics, _PATCH_FLAG, True)


def _missing_group_patch_current(diagnostics) -> bool:
    """Return whether advanced diagnostics still point to the missing-group wrappers."""

    return all(
        getattr(getattr(diagnostics, name, None), flag, False)
        for name, flag in (
            ("evidence_margin_table", _EVIDENCE_MARGIN_TABLE_WRAPPER_FLAG),
            ("add_evidence_margin_columns", _ADD_COLUMNS_WRAPPER_FLAG),
        )
    )
