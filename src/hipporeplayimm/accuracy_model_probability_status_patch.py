"""Normalize status values for accuracy-upgrade model-probability diagnostics."""

from __future__ import annotations

from functools import wraps

import pandas as pd

from .evidence_status_coercion import _normalize_status_value

_PATCHED_FLAG = "_model_probability_status_patch_applied"


def apply_model_probability_status_patch() -> None:
    """Install status normalization for accuracy-upgrade probability summaries."""

    from . import accuracy_upgrades

    if getattr(accuracy_upgrades, _PATCHED_FLAG, False):
        return

    original = accuracy_upgrades.model_probability_diagnostics

    @wraps(original)
    def model_probability_diagnostics(
        scores: pd.DataFrame,
        *,
        evidence_column: str = "log_evidence",
        group_columns=("session", "event_index"),
    ) -> pd.DataFrame:
        if scores.empty or "status" not in scores.columns:
            return original(
                scores,
                evidence_column=evidence_column,
                group_columns=group_columns,
            )
        normalized = scores.copy()
        normalized["status"] = normalized["status"].map(_normalize_status_value)
        return original(
            normalized,
            evidence_column=evidence_column,
            group_columns=group_columns,
        )

    accuracy_upgrades.model_probability_diagnostics = model_probability_diagnostics
    setattr(accuracy_upgrades, _PATCHED_FLAG, True)


__all__ = ["apply_model_probability_status_patch"]
