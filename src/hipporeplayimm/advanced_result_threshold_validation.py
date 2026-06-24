"""Validate paired model margin thresholds and neutral zero-margin ties."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_advanced_result_threshold_validation_patch_applied"


def _validated_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    values: list[float] = []
    for threshold in thresholds:
        value = float(threshold)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("thresholds must contain finite nonnegative values")
        values.append(value)
    return tuple(values)


def apply_advanced_result_threshold_validation_patch() -> None:
    """Install paired-threshold validation and neutral zero-margin tie handling."""

    from . import advanced_result_diagnostics as diagnostics

    if getattr(diagnostics, _PATCHED_FLAG, False):
        return

    previous_decisions = diagnostics.paired_model_margin_decisions
    previous_sweep = diagnostics.paired_model_margin_threshold_sweep

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
        out = previous_decisions(
            scores,
            positive_model=positive_model,
            reference_model=reference_model,
            margin_threshold=margin_threshold,
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
        return previous_sweep(
            scores,
            positive_model=positive_model,
            reference_model=reference_model,
            thresholds=_validated_thresholds(thresholds),
            group_cols=group_cols,
            evidence_col=evidence_col,
            model_col=model_col,
            true_model_col=true_model_col,
            positive_true_label=positive_true_label,
        )

    diagnostics.paired_model_margin_decisions = paired_model_margin_decisions
    diagnostics.paired_model_margin_threshold_sweep = paired_model_margin_threshold_sweep
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_advanced_result_threshold_validation_patch"]
