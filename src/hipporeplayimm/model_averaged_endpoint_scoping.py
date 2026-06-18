"""Patch model-averaged endpoint summaries to respect decode scope.

Improved evidence tables can contain several independent model-choice units for
the same ``(session, event_index)`` pair: replay-window variants, matched-null
windows, or cell-split benchmark repeats.  Endpoint averaging has to use the
same scope as evidence normalization; otherwise endpoint columns can mix distinct
windows even though their model probabilities were normalized separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_MODEL_AVERAGE_BASE_COLUMNS = ("session", "event_index")
_MODEL_AVERAGE_SCOPE_COLUMNS = (
    "benchmark_random_seed",
    "benchmark_cell_split_index",
    "benchmark_cell_split_seed",
    "benchmark_event_subset_seed",
    "benchmark_test_cell_fraction",
    "window_role",
    "window_index",
    "null_index",
    "matched_null_rank",
    "template_event_index",
    "event_window_variant",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
)


def apply_model_averaged_endpoint_scoping_patch() -> None:
    """Install the scoped endpoint-averaging implementation."""

    from . import result_improvement_extensions as extensions

    current = extensions.add_model_averaged_endpoint_columns
    if getattr(current, "_scoped_model_averaged_endpoints", False):
        return
    add_model_averaged_endpoint_columns._scoped_model_averaged_endpoints = True  # type: ignore[attr-defined]
    extensions.add_model_averaged_endpoint_columns = add_model_averaged_endpoint_columns


def add_model_averaged_endpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add scoped model-averaged endpoint estimates to an evidence table."""

    if df.empty or "model_probability" not in df:
        return df
    required = {"diagnostic_decoded_endpoint_x", "diagnostic_decoded_endpoint_y"}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    out["model_averaged_endpoint_x"] = np.nan
    out["model_averaged_endpoint_y"] = np.nan
    out["model_averaged_endpoint_models"] = 0
    out["model_probability_entropy"] = np.nan
    out["model_log_evidence_margin"] = np.nan

    group_columns = _model_average_group_columns(out)
    groups = [((), out)] if not group_columns else out.groupby(group_columns, sort=False, dropna=False)
    for _, group in groups:
        if "evidence_comparable" in group:
            comparable = _bool_series(group["evidence_comparable"])
        else:
            comparable = pd.Series(True, index=group.index)
        exact = group[comparable].copy()
        for column in (
            "model_probability",
            "diagnostic_decoded_endpoint_x",
            "diagnostic_decoded_endpoint_y",
            "log_evidence",
        ):
            if column in exact.columns:
                exact[column] = pd.to_numeric(exact[column], errors="coerce")
        exact = exact.dropna(
            subset=[
                "model_probability",
                "diagnostic_decoded_endpoint_x",
                "diagnostic_decoded_endpoint_y",
            ]
        )
        if exact.empty:
            continue

        weights = exact["model_probability"].to_numpy(dtype=float, copy=True)
        total = float(np.sum(weights))
        if total <= 0.0 or not np.isfinite(total):
            continue
        weights /= total

        x = float(np.sum(weights * exact["diagnostic_decoded_endpoint_x"].to_numpy(dtype=float)))
        y = float(np.sum(weights * exact["diagnostic_decoded_endpoint_y"].to_numpy(dtype=float)))
        positive = weights > 0.0
        entropy = float(-np.sum(weights[positive] * np.log(weights[positive])))
        if "log_evidence" in exact:
            logs = np.sort(exact["log_evidence"].to_numpy(dtype=float))[::-1]
            logs = logs[np.isfinite(logs)]
            margin = float(logs[0] - logs[1]) if logs.size > 1 else np.inf
        else:
            margin = np.nan

        out.loc[group.index, "model_averaged_endpoint_x"] = x
        out.loc[group.index, "model_averaged_endpoint_y"] = y
        out.loc[group.index, "model_averaged_endpoint_models"] = int(exact.shape[0])
        out.loc[group.index, "model_probability_entropy"] = entropy
        out.loc[group.index, "model_log_evidence_margin"] = margin
    return out


def _model_average_group_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns identifying one independent model-choice scope."""

    columns = [column for column in _MODEL_AVERAGE_BASE_COLUMNS if column in frame.columns]
    for column in _MODEL_AVERAGE_SCOPE_COLUMNS:
        if column in frame.columns and column not in columns:
            columns.append(column)
    return columns


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "t", "yes", "y", "on"}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_bool_value).astype(bool)
