"""Retain rows with missing grouping provenance in grouped model summaries."""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import pandas as pd

_PATCHED_FLAG = "_grouped_model_metrics_missing_group_patch_applied"
_WRAPPER_FLAG = "_grouped_model_metrics_missing_group_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_grouped_metrics_original__"
_SENTINEL_BASE = "__hipporeplayimm_missing_group__"


def apply_grouped_model_metrics_missing_group_patch() -> None:
    """Keep missing session/model keys visible in grouped metric summaries."""

    from . import result_improvements

    current = result_improvements.summarize_grouped_model_metrics
    if getattr(current, _WRAPPER_FLAG, False):
        setattr(result_improvements, _PATCHED_FLAG, True)
        return

    original = current

    @wraps(original)
    def summarize_grouped_model_metrics(
        rows: pd.DataFrame,
        group_columns: tuple[str, ...],
        *,
        value_columns: tuple[str, ...] = (
            "heldout_log_likelihood",
            "delta_vs_best_static",
            "bits_per_spike_vs_best_static",
            "lower_bound_delta_vs_best_static",
            "lower_bound_bits_per_spike_vs_best_static",
        ),
    ) -> pd.DataFrame:
        prepared = rows
        sentinels: dict[str, str] = {}
        for column in (*group_columns, "model"):
            if column not in rows:
                continue
            missing = rows[column].isna()
            if not bool(missing.any()):
                continue
            if prepared is rows:
                prepared = rows.copy()
            sentinel = _missing_group_sentinel(rows[column], column)
            prepared[column] = prepared[column].astype(object)
            prepared.loc[missing, column] = sentinel
            sentinels[column] = sentinel

        summary = original(
            prepared,
            group_columns,
            value_columns=value_columns,
        )
        if summary.empty or not sentinels:
            return summary

        restored = summary.copy()
        for column, sentinel in sentinels.items():
            if column not in restored:
                continue
            matches = restored[column].astype(object).eq(sentinel)
            restored.loc[matches, column] = pd.NA
        return restored

    setattr(summarize_grouped_model_metrics, _WRAPPER_FLAG, True)
    setattr(summarize_grouped_model_metrics, _ORIGINAL_ATTR, original)
    result_improvements.summarize_grouped_model_metrics = summarize_grouped_model_metrics
    _synchronize_aliases(original, summarize_grouped_model_metrics)
    setattr(result_improvements, _PATCHED_FLAG, True)


def _missing_group_sentinel(values: pd.Series, column: str) -> str:
    """Return a private label that cannot collide with observed group values."""

    observed = {str(value) for value in values.dropna().tolist()}
    base = f"{_SENTINEL_BASE}{column}__"
    sentinel = base
    suffix = 1
    while sentinel in observed:
        sentinel = f"{base}_{suffix}"
        suffix += 1
    return sentinel


def _synchronize_aliases(original: Any, patched: Any) -> None:
    """Refresh package-local imports of the grouped summary helper."""

    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("hipporeplayimm"):
            continue
        if getattr(module, "summarize_grouped_model_metrics", None) is original:
            module.summarize_grouped_model_metrics = patched


__all__ = ["apply_grouped_model_metrics_missing_group_patch"]
