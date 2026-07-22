"""Keep evidence-margin diagnostics finite and distinct by model."""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_evidence_margin_distinct_model_patch_applied"
_MARGIN_FLAG = "_evidence_margin_distinct_model_wrapper"
_ADD_FLAG = "_evidence_margin_distinct_model_add_columns_wrapper"
_BOOL_FLAG = "_evidence_margin_arbitrary_integer_bool_wrapper"
_EMPTY_SWEEP_FLAG = "_evidence_margin_empty_threshold_sweep_wrapper"
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
    current_sweep = diagnostics.paired_model_margin_threshold_sweep
    if (
        getattr(diagnostics, _PATCHED_FLAG, False)
        and getattr(current_margin, _MARGIN_FLAG, False)
        and getattr(current_add, _ADD_FLAG, False)
        and getattr(current_bool, _BOOL_FLAG, False)
        and getattr(current_sweep, _EMPTY_SWEEP_FLAG, False)
    ):
        return
    # The threshold-validation patch replaces the sweep wrapper during a runtime
    # refresh.  That makes this patch run again even though its margin and bool
    # wrappers are still installed.  Reuse their stored bases instead of wrapping
    # our own wrappers repeatedly.
    original_margin = _original_before_refresh(current_margin, _MARGIN_FLAG)
    original_bool = _original_before_refresh(current_bool, _BOOL_FLAG)
    original_sweep = _original_before_refresh(current_sweep, _EMPTY_SWEEP_FLAG)

    def evidence_margin_table(
        scores,
        *,
        group_cols=_DEFAULT_GROUP_COLUMNS,
        evidence_col="log_evidence",
        model_col="model",
    ):
        groups = _normalize_group_cols(group_cols)
        collapsed = _collapse_duplicate_models(
            diagnostics, scores, groups, evidence_col, model_col
        )
        target = scores if collapsed is None else collapsed
        if groups:
            return original_margin(
                target,
                group_cols=groups,
                evidence_col=evidence_col,
                model_col=model_col,
            )
        return _global_evidence_margin_table(
            original_margin,
            target,
            evidence_col=evidence_col,
            model_col=model_col,
        )

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
        if isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        ):
            return int(value) != 0
        return original_bool(value, default=default)

    @wraps(original_sweep)
    def paired_model_margin_threshold_sweep(
        scores: pd.DataFrame,
        *,
        positive_model: str,
        reference_model: str,
        thresholds: Sequence[float],
        group_cols: Sequence[str] | str | None = None,
        evidence_col: str = "log_evidence",
        model_col: str = "model",
        true_model_col: str | None = None,
        positive_true_label: str | None = None,
    ) -> pd.DataFrame:
        """Preserve the public threshold-sweep schema for zero thresholds."""

        out = original_sweep(
            scores,
            positive_model=positive_model,
            reference_model=reference_model,
            thresholds=thresholds,
            group_cols=group_cols,
            evidence_col=evidence_col,
            model_col=model_col,
            true_model_col=true_model_col,
            positive_true_label=positive_true_label,
        )
        if not out.empty or len(out.columns) > 0:
            return out

        if group_cols is None:
            paired_group_cols = tuple(
                diagnostics.infer_paired_model_group_cols(scores)
            )
        elif isinstance(group_cols, str):
            paired_group_cols = (group_cols,)
        else:
            paired_group_cols = tuple(group_cols)

        summary = diagnostics.paired_model_margin_summary(
            pd.DataFrame(),
            true_model_col=true_model_col,
        ).copy()
        summary["positive_model"] = str(positive_model)
        summary["reference_model"] = str(reference_model)
        summary["margin_threshold"] = np.nan
        if true_model_col:
            defaults: dict[str, float | int] = {
                "thresholded_binary_accuracy": np.nan,
                "positive_true_events": 0,
                "reference_true_events": 0,
                "positive_true_claimed_events": 0,
                "reference_true_rejected_events": 0,
                "positive_claim_recall": np.nan,
                "reference_specificity": np.nan,
                "false_positive_claims": 0,
                "false_negative_claims": 0,
            }
            for column, value in defaults.items():
                if column not in summary.columns:
                    summary[column] = value
        summary["group_cols"] = ",".join(paired_group_cols)
        return summary.iloc[0:0].copy()

    setattr(evidence_margin_table, _MARGIN_FLAG, True)
    setattr(evidence_margin_table, "__hipporeplayimm_original__", original_margin)
    setattr(add_evidence_margin_columns, _ADD_FLAG, True)
    setattr(_as_bool, _BOOL_FLAG, True)
    setattr(_as_bool, "__hipporeplayimm_original__", original_bool)
    setattr(paired_model_margin_threshold_sweep, _EMPTY_SWEEP_FLAG, True)
    setattr(
        paired_model_margin_threshold_sweep,
        "__hipporeplayimm_original__",
        original_sweep,
    )
    diagnostics.evidence_margin_table = evidence_margin_table
    diagnostics.add_evidence_margin_columns = add_evidence_margin_columns
    diagnostics._as_bool = _as_bool
    diagnostics.paired_model_margin_threshold_sweep = (
        paired_model_margin_threshold_sweep
    )
    setattr(diagnostics, _PATCHED_FLAG, True)


def _original_before_refresh(function, wrapper_flag: str):
    """Return the base below this patch's current wrapper, when present."""

    if not getattr(function, wrapper_flag, False):
        return function
    return getattr(function, "__hipporeplayimm_original__", function)


def _collapse_duplicate_models(
    diagnostics,
    scores: pd.DataFrame,
    group_cols: tuple[str, ...],
    evidence_col: str,
    model_col: str,
) -> pd.DataFrame | None:
    if scores.empty:
        return None
    ok = diagnostics._comparable_rows(scores)
    if ok.empty:
        return ok
    if any(
        column not in ok.columns
        for column in (*group_cols, evidence_col, model_col)
    ):
        return None
    rows = []
    grouped = (
        ok.groupby(list(group_cols), sort=False, dropna=False)
        if group_cols
        else (((), ok),)
    )
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


def _global_evidence_margin_table(
    original_margin,
    scores: pd.DataFrame,
    *,
    evidence_col: str,
    model_col: str,
) -> pd.DataFrame:
    """Evaluate one table-wide evidence margin without calling ``groupby([])``."""

    group_column = "__hipporeplayimm_global_evidence_margin_group__"
    while group_column in scores.columns:
        group_column += "_"
    global_scores = scores.copy()
    global_scores[group_column] = 0
    margins = original_margin(
        global_scores,
        group_cols=(group_column,),
        evidence_col=evidence_col,
        model_col=model_col,
    )
    return margins.drop(columns=[group_column], errors="ignore")


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

    if not group_cols:
        if len(margins) != 1:
            raise ValueError("global evidence margins must contain exactly one row")
        out = scores.copy()
        margin = margins.iloc[0]
        for column in margins.columns:
            out[column] = margin[column]
        return out
    row_position_column = "__hipporeplayimm_margin_row_position__"
    while (
        row_position_column in scores.columns
        or row_position_column in margins.columns
    ):
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
