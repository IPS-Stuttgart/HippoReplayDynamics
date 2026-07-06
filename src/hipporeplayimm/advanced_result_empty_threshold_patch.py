"""Preserve paired-threshold sweep schemas when no complete model pairs exist.

Advanced paired-model threshold sweeps may be run on filtered diagnostic tables
where every group is missing either the positive or reference model.  The
underlying decision table is then empty but still has a meaningful threshold and
model-pair context.  Keep those columns present so downstream threshold
selection reports a visible fallback instead of failing with a missing-column
error.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_advanced_result_empty_threshold_patch_applied"


def _normalize_group_cols(group_cols: Sequence[str] | str | None, scores: pd.DataFrame) -> tuple[str, ...]:
    if group_cols is None:
        from . import advanced_result_diagnostics as diagnostics

        return tuple(diagnostics.infer_paired_model_group_cols(scores))
    if isinstance(group_cols, str):
        return (group_cols,)
    return tuple(group_cols)


def _with_threshold_context(
    summary: pd.DataFrame,
    *,
    positive_model: str,
    reference_model: str,
    threshold: float,
    true_model_col: str | None,
) -> pd.DataFrame:
    """Attach model-pair and gate columns to one threshold-summary row."""

    out = summary.copy()
    out["positive_model"] = str(positive_model)
    out["reference_model"] = str(reference_model)
    out["margin_threshold"] = float(threshold)

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
            if column not in out:
                out[column] = value
    return out


def _apply_distinct_model_margin_patch() -> None:
    """Install the event-margin duplicate-model fix from package startup."""

    from . import advanced_result_margin_duplicate_patch

    advanced_result_margin_duplicate_patch.apply_advanced_result_margin_duplicate_patch()


def apply_advanced_result_empty_threshold_patch() -> None:
    """Install empty-pair threshold-sweep schema preservation."""

    from . import advanced_result_diagnostics as diagnostics

    _apply_distinct_model_margin_patch()
    if getattr(diagnostics, _PATCHED_FLAG, False):
        return

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
        """Summarize paired margin decisions over candidate thresholds."""

        paired_group_cols = _normalize_group_cols(group_cols, scores)
        rows: list[pd.DataFrame] = []
        for threshold in thresholds:
            threshold_value = float(threshold)
            decisions = diagnostics.paired_model_margin_decisions(
                scores,
                positive_model=positive_model,
                reference_model=reference_model,
                margin_threshold=threshold_value,
                group_cols=paired_group_cols,
                evidence_col=evidence_col,
                model_col=model_col,
                true_model_col=true_model_col,
                positive_true_label=positive_true_label,
            )
            summary = diagnostics.paired_model_margin_summary(
                decisions,
                true_model_col=true_model_col,
            )
            summary = _with_threshold_context(
                summary,
                positive_model=positive_model,
                reference_model=reference_model,
                threshold=threshold_value,
                true_model_col=true_model_col,
            )
            summary["group_cols"] = ",".join(paired_group_cols)
            rows.append(summary)
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, ignore_index=True)
        return out.sort_values("margin_threshold", kind="stable").reset_index(drop=True)

    diagnostics.paired_model_margin_threshold_sweep = paired_model_margin_threshold_sweep
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_advanced_result_empty_threshold_patch"]
