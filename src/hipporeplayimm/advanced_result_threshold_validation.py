"""Reject non-finite paired model margin thresholds."""

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
    """Install finite nonnegative validation for paired threshold sweeps."""

    from . import advanced_result_diagnostics as diagnostics

    if getattr(diagnostics, _PATCHED_FLAG, False):
        return

    previous = diagnostics.paired_model_margin_threshold_sweep

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
        return previous(
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

    diagnostics.paired_model_margin_threshold_sweep = paired_model_margin_threshold_sweep
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_advanced_result_threshold_validation_patch"]
