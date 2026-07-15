"""Patch model-averaged endpoint summaries to respect decode scope."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmark_relative_grouping import _BENCHMARK_RELATIVE_SCOPE_COLUMNS

_MODEL_AVERAGE_BASE_COLUMNS = ("session", "event_index")
_MODEL_AVERAGE_SCOPE_COLUMNS = (
    "benchmark_random_seed",
    "random_seed",
    "null_random_seed",
    "benchmark_cell_split_index",
    *_BENCHMARK_RELATIVE_SCOPE_COLUMNS,
    "train_cell_ids",
    "test_cell_ids",
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
    from . import result_improvement_extensions as extensions

    current = extensions.add_model_averaged_endpoint_columns
    if getattr(current, "_scoped_model_averaged_endpoints", False):
        return
    add_model_averaged_endpoint_columns._scoped_model_averaged_endpoints = True  # type: ignore[attr-defined]
    extensions.add_model_averaged_endpoint_columns = add_model_averaged_endpoint_columns


def add_model_averaged_endpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    for row_positions, group in _model_average_groups(out):
        comparable = _bool_series(group["evidence_comparable"]) if "evidence_comparable" in group else pd.Series(True, index=group.index)
        exact = group[comparable].copy()
        for column in (
            "model_probability",
            "diagnostic_decoded_endpoint_x",
            "diagnostic_decoded_endpoint_y",
            "log_evidence",
        ):
            if column in exact.columns:
                exact[column] = pd.to_numeric(exact[column], errors="coerce")
        exact = exact.dropna(subset=["model_probability", "diagnostic_decoded_endpoint_x", "diagnostic_decoded_endpoint_y"])
        exact = _finite_endpoint_average_rows(exact)
        exact = _distinct_model_rows(exact)
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
        margin = _log_evidence_margin(exact)

        positions = np.asarray(row_positions, dtype=int)
        out.iloc[positions, out.columns.get_loc("model_averaged_endpoint_x")] = x
        out.iloc[positions, out.columns.get_loc("model_averaged_endpoint_y")] = y
        out.iloc[positions, out.columns.get_loc("model_averaged_endpoint_models")] = int(exact.shape[0])
        out.iloc[positions, out.columns.get_loc("model_probability_entropy")] = entropy
        out.iloc[positions, out.columns.get_loc("model_log_evidence_margin")] = margin
    return out


def _distinct_model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the strongest finite-evidence row for each model identity."""

    if frame.empty or "model" not in frame.columns:
        return frame
    if "log_evidence" not in frame.columns:
        return frame.drop_duplicates("model", keep="first").copy()

    evidence = pd.to_numeric(frame["log_evidence"], errors="coerce").to_numpy(dtype=float)
    sort_key = np.where(np.isfinite(evidence), -evidence, np.inf)
    order = np.argsort(sort_key, kind="stable")
    return frame.iloc[order].drop_duplicates("model", keep="first").copy()


def _log_evidence_margin(exact: pd.DataFrame) -> float:
    if "log_evidence" not in exact:
        return np.nan
    logs = np.sort(exact["log_evidence"].to_numpy(dtype=float))[::-1]
    logs = logs[np.isfinite(logs)]
    if logs.size > 1:
        return float(logs[0] - logs[1])
    return np.nan


def _finite_endpoint_average_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    finite = pd.Series(True, index=frame.index)
    for column in ("model_probability", "diagnostic_decoded_endpoint_x", "diagnostic_decoded_endpoint_y"):
        values = pd.to_numeric(frame[column], errors="coerce")
        finite &= np.isfinite(values.to_numpy(dtype=float))
    probabilities = pd.to_numeric(frame["model_probability"], errors="coerce")
    finite &= probabilities >= 0.0
    return frame.loc[finite].copy()


def _model_average_groups(frame: pd.DataFrame):
    group_columns = _model_average_group_columns(frame)
    if not group_columns:
        row_positions = np.arange(len(frame), dtype=int)
        yield row_positions, frame.iloc[row_positions]
        return
    labels = pd.DataFrame(
        {
            f"__model_average_scope_{index}": [_scope_label(value) for value in frame[column]]
            for index, column in enumerate(group_columns)
        },
        index=frame.index,
    )
    for indices in labels.groupby(list(labels.columns), sort=False, dropna=False).indices.values():
        row_positions = np.asarray(indices, dtype=int)
        yield row_positions, frame.iloc[row_positions]


def _model_average_group_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in _MODEL_AVERAGE_BASE_COLUMNS if column in frame.columns]
    for column in _MODEL_AVERAGE_SCOPE_COLUMNS:
        if column in frame.columns and column not in columns:
            columns.append(column)
    return columns


def _scope_label(value: object) -> str:
    if _is_missing_scalar(value):
        return "<missing>"
    if isinstance(value, (bool, np.bool_)):
        return repr(("scalar", str(bool(value))))
    if isinstance(value, (int, np.integer)):
        return repr(("numeric", int(value)))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric):
            return repr(("numeric", int(numeric) if numeric.is_integer() else numeric))
    if isinstance(value, np.ndarray):
        return repr(("array", np.asarray(value, dtype=object).reshape(-1).tolist()))
    if isinstance(value, (list, tuple)):
        return repr(("sequence", list(value)))
    if isinstance(value, set):
        return repr(("set", sorted(value, key=repr)))
    return repr(("scalar", str(value).strip()))


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "on"}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_bool_value).astype(bool)
