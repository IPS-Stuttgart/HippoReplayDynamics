"""Preserve advanced diagnostics groups with missing scope metadata."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .wrong_map_missing_group_patch import apply_wrong_map_missing_group_patch, wrong_map_missing_group_patch_current

_PATCH_FLAG = "_missing_group_metadata_patch_applied"
_EVIDENCE_MARGIN_TABLE_WRAPPER_FLAG = "_missing_group_metadata_evidence_margin_table_wrapper"
_ADD_COLUMNS_WRAPPER_FLAG = "_missing_group_metadata_add_margin_columns_wrapper"
_PAIRED_DECISIONS_WRAPPER_FLAG = "_missing_group_metadata_paired_margin_decisions_wrapper"
_THRESHOLD_BASE_DECISIONS_ATTR = "_advanced_result_threshold_validation_base_decisions"
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
    """Keep diagnostics for rows whose optional grouping metadata is missing."""

    from . import advanced_result_diagnostics as diagnostics
    from . import advanced_result_threshold_validation as threshold_validation

    if (
        getattr(diagnostics, _PATCH_FLAG, False)
        and _missing_group_patch_current(diagnostics)
        and _paired_missing_group_patch_current(diagnostics)
        and wrong_map_missing_group_patch_current(diagnostics)
    ):
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
        for column in group_cols:
            if column in scores.columns and column in margins.columns:
                margins[column] = margins[column].astype(scores[column].dtype)
        return scores.merge(margins, on=list(group_cols), how="left")

    def paired_model_margin_decisions(
        scores: pd.DataFrame,
        *,
        positive_model: str,
        reference_model: str,
        margin_threshold: float = 0.0,
        group_cols: Sequence[str] = ("session", "event_index"),
        evidence_col: str = "log_evidence",
        model_col: str = "model",
        true_model_col: str | None = None,
        positive_true_label: str | None = None,
    ) -> pd.DataFrame:
        """Classify paired model wins, retaining NA event-group keys."""

        threshold = float(margin_threshold)
        if threshold < 0.0:
            raise ValueError("margin_threshold must be non-negative")
        ok = diagnostics._comparable_rows(scores)
        columns = [
            *group_cols,
            "positive_model",
            "reference_model",
            "positive_log_evidence",
            "reference_log_evidence",
            "positive_minus_reference_log_evidence",
            "margin_threshold",
            "margin_decision",
            "positive_model_claimed",
        ]
        if true_model_col:
            columns.extend(
                [
                    true_model_col,
                    "positive_true_label",
                    "true_is_positive",
                    "margin_binary_correct",
                ]
            )
        if ok.empty:
            return pd.DataFrame(columns=columns)
        missing = [column for column in (*group_cols, evidence_col, model_col) if column not in ok.columns]
        if true_model_col and true_model_col not in ok.columns:
            missing.append(true_model_col)
        if missing:
            raise KeyError(f"scores is missing required columns: {missing}")

        positive_label = positive_true_label or diagnostics._model_family_label(positive_model)
        rows: list[dict[str, object]] = []
        for key, group in ok.groupby(list(group_cols), sort=False, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            paired = group[group[model_col].astype(str).isin([positive_model, reference_model])]
            pivot = paired.dropna(subset=[evidence_col]).drop_duplicates(model_col, keep="last")
            by_model = pivot.set_index(model_col)
            if positive_model not in by_model.index or reference_model not in by_model.index:
                continue
            positive_value = float(by_model.loc[positive_model, evidence_col])
            reference_value = float(by_model.loc[reference_model, evidence_col])
            delta = positive_value - reference_value
            if delta >= threshold:
                decision = positive_model
                positive_claimed = True
            elif delta <= -threshold:
                decision = reference_model
                positive_claimed = False
            else:
                decision = "ambiguous"
                positive_claimed = False
            row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
            row.update(
                {
                    "positive_model": positive_model,
                    "reference_model": reference_model,
                    "positive_log_evidence": positive_value,
                    "reference_log_evidence": reference_value,
                    "positive_minus_reference_log_evidence": float(delta),
                    "margin_threshold": threshold,
                    "margin_decision": decision,
                    "positive_model_claimed": bool(positive_claimed),
                }
            )
            if true_model_col:
                true_label = diagnostics._unique_text_value(group[true_model_col])
                true_is_positive = diagnostics._model_family_label(true_label) == diagnostics._model_family_label(positive_label)
                row.update(
                    {
                        true_model_col: true_label,
                        "positive_true_label": positive_label,
                        "true_is_positive": bool(true_is_positive),
                        "margin_binary_correct": bool(positive_claimed) == bool(true_is_positive),
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    setattr(evidence_margin_table, _EVIDENCE_MARGIN_TABLE_WRAPPER_FLAG, True)
    setattr(add_evidence_margin_columns, _ADD_COLUMNS_WRAPPER_FLAG, True)
    setattr(paired_model_margin_decisions, _PAIRED_DECISIONS_WRAPPER_FLAG, True)
    diagnostics.evidence_margin_table = evidence_margin_table
    diagnostics.add_evidence_margin_columns = add_evidence_margin_columns
    diagnostics.paired_model_margin_decisions = paired_model_margin_decisions
    setattr(diagnostics, _THRESHOLD_BASE_DECISIONS_ATTR, paired_model_margin_decisions)
    threshold_validation.apply_advanced_result_threshold_validation_patch()
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


def _paired_missing_group_patch_current(diagnostics) -> bool:
    """Return whether paired decisions still use the NA-preserving base wrapper."""

    base_decisions = getattr(diagnostics, _THRESHOLD_BASE_DECISIONS_ATTR, None)
    return bool(getattr(base_decisions, _PAIRED_DECISIONS_WRAPPER_FLAG, False))
