"""Patch evidence-margin diagnostics to compare distinct model alternatives."""

from __future__ import annotations

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_advanced_result_margin_duplicate_patch_applied"
_WRAPPER_FLAG = "_advanced_result_margin_distinct_models_wrapper"


def apply_advanced_result_margin_duplicate_patch() -> None:
    """Install duplicate-model handling for event-level evidence margins."""

    from . import advanced_result_diagnostics as diagnostics

    current = diagnostics.evidence_margin_table
    if getattr(current, _WRAPPER_FLAG, False):
        setattr(diagnostics, _PATCHED_FLAG, True)
        return

    def evidence_margin_table(
        scores: pd.DataFrame,
        *,
        group_cols=("session", "event_index"),
        evidence_col: str = "log_evidence",
        model_col: str = "model",
    ) -> pd.DataFrame:
        """Return one calibrated margin row using distinct model identities."""

        ok = diagnostics._comparable_rows(scores)
        columns = [
            *group_cols,
            "best_model_by_evidence",
            "second_best_model_by_evidence",
            "best_log_evidence",
            "second_best_log_evidence",
            "evidence_margin_to_second_best",
            "evidence_margin_category",
            "models_compared",
        ]
        if ok.empty:
            return pd.DataFrame(columns=columns)
        missing = [column for column in (*group_cols, evidence_col, model_col) if column not in ok.columns]
        if missing:
            raise KeyError(f"scores is missing required columns: {missing}")

        rows: list[dict[str, object]] = []
        for key, group in ok.groupby(list(group_cols), sort=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            group = (
                _finite_numeric_evidence_rows(group, evidence_col)
                .sort_values(evidence_col, ascending=False, kind="stable")
                .drop_duplicates(model_col, keep="first")
            )
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
        return pd.DataFrame(rows, columns=columns)

    setattr(evidence_margin_table, _WRAPPER_FLAG, True)
    diagnostics.evidence_margin_table = evidence_margin_table
    setattr(diagnostics, _PATCHED_FLAG, True)


def _finite_numeric_evidence_rows(frame: pd.DataFrame, evidence_col: str) -> pd.DataFrame:
    """Return rows whose evidence column is finite after numeric CSV coercion."""

    out = frame.copy()
    evidence = pd.to_numeric(out[evidence_col], errors="coerce")
    finite = np.isfinite(evidence.to_numpy(dtype=float))
    out = out.loc[finite].copy()
    out[evidence_col] = evidence.loc[out.index].astype(float)
    return out


__all__ = ["apply_advanced_result_margin_duplicate_patch"]
