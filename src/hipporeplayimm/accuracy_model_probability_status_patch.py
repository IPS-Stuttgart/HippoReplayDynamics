"""Normalize status/evidence values for accuracy-upgrade model-probability diagnostics."""

from __future__ import annotations

from functools import wraps

import pandas as pd

from .evidence_status_coercion import _normalize_status_value

_PATCHED_FLAG = "_model_probability_status_patch_applied"


def apply_model_probability_status_patch() -> None:
    """Install input normalization for accuracy-upgrade probability summaries."""

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
        normalized = scores
        if not scores.empty and ("status" in scores.columns or evidence_column in scores.columns):
            normalized = scores.copy()
            if "status" in normalized.columns:
                normalized["status"] = normalized["status"].map(_normalize_status_value)
            if evidence_column in normalized.columns:
                normalized[evidence_column] = pd.to_numeric(normalized[evidence_column], errors="coerce")
                normalized = normalized.dropna(subset=[evidence_column])
        return original(
            normalized,
            evidence_column=evidence_column,
            group_columns=group_columns,
        )

    accuracy_upgrades.model_probability_diagnostics = model_probability_diagnostics
    setattr(accuracy_upgrades, _PATCHED_FLAG, True)


__all__ = ["apply_model_probability_status_patch"]
