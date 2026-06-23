"""Keep legacy missing-status rows in advanced result diagnostics.

Older score-table artifacts can contain a ``status`` column whose successful rows
round-trip through CSV as blanks or nulls.  The advanced diagnostics used a
literal ``status == 'success'`` filter, which dropped those legacy rows before
margin, wrong-map, and paired-model summaries were computed.
"""

from __future__ import annotations

import pandas as pd

from .evidence_status_coercion import _status_is_success_or_missing

_PATCHED_FLAG = "_advanced_result_status_patch_applied"


def apply_advanced_result_status_patch() -> None:
    """Patch advanced diagnostics to treat missing legacy statuses as success."""

    from . import advanced_result_diagnostics as diagnostics

    if getattr(diagnostics, _PATCHED_FLAG, False):
        return

    def successful_rows(scores: pd.DataFrame) -> pd.DataFrame:
        if scores.empty:
            return scores.copy()
        if "status" not in scores.columns:
            return scores.copy()
        mask = scores["status"].map(_status_is_success_or_missing).astype(bool)
        return scores[mask].copy()

    diagnostics._successful_rows = successful_rows
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_advanced_result_status_patch"]
