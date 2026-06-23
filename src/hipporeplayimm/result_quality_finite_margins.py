"""Ignore non-finite evidence values in result-quality margin rankings.

Quality summaries rank exact model evidences and truncated lower-bound diagnostics
separately. Non-finite log-evidence values cannot define a meaningful best model
or margin in either scope, so they should be excluded before ranking rather than
being allowed to dominate with ``inf`` or produce ``nan`` margins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_result_quality_finite_margins_patch_applied"


def apply_result_quality_finite_margins_patch() -> None:
    """Patch result-quality margin annotation to require finite evidence."""

    from . import result_quality_gates as gates

    if getattr(gates, _PATCHED_FLAG, False):
        return

    def annotate_margin_scope(
        out: pd.DataFrame,
        group_index: pd.Index,
        rows: pd.DataFrame,
        *,
        prefix: str,
    ) -> None:
        if rows.empty or "log_evidence" not in rows:
            return
        values = pd.to_numeric(rows["log_evidence"], errors="coerce")
        finite = pd.Series(
            np.isfinite(values.to_numpy(dtype=float)),
            index=values.index,
        )
        rows = rows.loc[finite].copy()
        values = values.loc[rows.index].to_numpy(dtype=float)
        if rows.empty:
            return
        order = np.argsort(-values, kind="mergesort")
        ordered_index = rows.index.to_numpy()[order]
        ordered_values = values[order]
        best_index = ordered_index[0]
        best_model = str(out.loc[best_index, "model"]) if "model" in out else ""
        margin = float(ordered_values[0] - ordered_values[1]) if ordered_values.shape[0] > 1 else np.inf
        out.loc[group_index, f"{prefix}_best_model"] = best_model
        out.loc[group_index, f"{prefix}_log_evidence_margin"] = margin
        out.loc[group_index, f"{prefix}_margin_category"] = gates.evidence_margin_label(margin)
        out.loc[ordered_index, f"{prefix}_rank"] = np.arange(1, ordered_index.shape[0] + 1, dtype=float)
        out.loc[ordered_index, f"{prefix}_relative_log_evidence"] = ordered_values - ordered_values[0]

    gates._annotate_margin_scope = annotate_margin_scope
    setattr(gates, _PATCHED_FLAG, True)


__all__ = ["apply_result_quality_finite_margins_patch"]
