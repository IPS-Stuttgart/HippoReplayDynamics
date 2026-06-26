"""Validate paired model margin thresholds and neutral zero-margin ties."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_advanced_result_threshold_validation_patch_applied"
_BASE_DECISIONS_ATTR = "_advanced_result_threshold_validation_base_decisions"
_PATCHED_DECISIONS_ATTR = "_advanced_result_threshold_validation_patched_decisions"
_PATCHED_SWEEP_ATTR = "_advanced_result_threshold_validation_patched_sweep"


def _validated_threshold(
    threshold: float,
    *,
    message: str = "margin_threshold must be a finite nonnegative value",
) -> float:
    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError(message)
    try:
        value = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(message)
    return value


def _validated_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        _validated_threshold(
            threshold,
            message="thresholds must contain finite nonnegative values",
        )
        for threshold in thresholds
    )


def _ensure_true_model_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep threshold-selection columns present even when no paired events exist."""

    defaults: dict[str, object] = {
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
    out = summary.copy()
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
    return out


def _current_patch_installed(diagnostics) -> bool:
    return (
        getattr(diagnostics, _PATCHED_FLAG, False)
        and getattr(diagnostics, "paired_model_margin_decisions", None) is getattr(diagnostics, _PATCHED_DECISIONS_ATTR, None)
        and getattr(diagnostics, "paired_model_margin_threshold_sweep", None) is getattr(diagnostics, _PATCHED_SWEEP_ATTR, None)
    )


def apply_advanced_result_threshold_validation_patch() -> None:
    """Install paired-threshold validation and neutral zero-margin tie handling."""

    from . import advanced_result_diagnostics as diagnostics

    if _current_patch_installed(diagnostics):
        return

    base_decisions = getattr(diagnostics, _BASE_DECISIONS_ATTR, None)
    if base_decisions is None:
        base_decisions = diagnostics.paired_model_margin_decisions
        setattr(diagnostics, _BASE_DECISIONS_ATTR, base_decisions)

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
        threshold = _validated_threshold(margin_threshold)
        out = base_decisions(
            scores,
            positive_model=positive_model,
            reference_model=reference_model,
            margin_threshold=threshold,
            group_cols=group_cols,
            evidence_col=evidence_col,
            model_col=model_col,
            true_model_col=true_model_col,
            positive_true_label=positive_true_label,
        )
        if out.empty:
            return out
        zero_tie = np.isclose(
            pd.to_numeric(out["margin_threshold"], errors="coerce"),
            0.0,
        ) & np.isclose(
            pd.to_numeric(out["positive_minus_reference_log_evidence"], errors="coerce"),
            0.0,
        )
        if not bool(zero_tie.any()):
            return out
        out = out.copy()
        out.loc[zero_tie, "margin_decision"] = "ambiguous"
        out.loc[zero_tie, "positive_model_claimed"] = False
        if "margin_binary_correct" in out and "true_is_positive" in out:
            true_is_positive = diagnostics._bool_column(out, "true_is_positive")
            out.loc[zero_tie, "margin_binary_correct"] = (~true_is_positive.loc[zero_tie]).astype(bool)
        return out

    def paired_model_margin_threshold_sweep(
        scores: pd.DataFrame,
        *,
        positive_model: str,
        reference_model: str,
        thresholds: Sequence[float],
        group_cols: Sequence[str] | None = None,
        evidence_col: str = "log_evidence",
        model_col: str = "model",
        true_model_col: str | None = None,
        positive_true_label: str | None = None,
    ) -> pd.DataFrame:
        paired_group_cols = tuple(group_cols) if group_cols is not None else diagnostics.infer_paired_model_group_cols(scores)
        rows: list[pd.DataFrame] = []
        for threshold in _validated_thresholds(thresholds):
            decisions = paired_model_margin_decisions(
                scores,
                positive_model=positive_model,
                reference_model=reference_model,
                margin_threshold=threshold,
                group_cols=paired_group_cols,
                evidence_col=evidence_col,
                model_col=model_col,
                true_model_col=true_model_col,
                positive_true_label=positive_true_label,
            )
            summary = diagnostics.paired_model_margin_summary(decisions, true_model_col=true_model_col).copy()
            if true_model_col:
                summary = _ensure_true_model_summary_columns(summary)
            summary["positive_model"] = str(positive_model)
            summary["reference_model"] = str(reference_model)
            summary["margin_threshold"] = float(threshold)
            summary["group_cols"] = ",".join(paired_group_cols)
            rows.append(summary)
        if not rows:
            return pd.DataFrame()
        out = pd.concat(rows, ignore_index=True)
        return out.sort_values("margin_threshold", kind="stable").reset_index(drop=True)

    diagnostics.paired_model_margin_decisions = paired_model_margin_decisions
    diagnostics.paired_model_margin_threshold_sweep = paired_model_margin_threshold_sweep
    setattr(diagnostics, _PATCHED_DECISIONS_ATTR, paired_model_margin_decisions)
    setattr(diagnostics, _PATCHED_SWEEP_ATTR, paired_model_margin_threshold_sweep)
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_advanced_result_threshold_validation_patch"]
