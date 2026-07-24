"""Patch model-averaged endpoint summaries to respect decode scope."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .benchmark_relative_grouping import _BENCHMARK_RELATIVE_SCOPE_COLUMNS

_MODEL_AVERAGE_BASE_COLUMNS = ("session", "event_index")
_MODEL_AVERAGE_SCOPE_COLUMNS = (
    "event_id",
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
_TRUE_BOOL_TEXT_VALUES = {"1", "1.0", "true", "t", "yes", "y", "on"}


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
        scale = float(np.max(weights))
        if scale <= 0.0 or not np.isfinite(scale):
            continue
        weights /= scale
        total = float(np.sum(weights))
        if total <= 0.0 or not np.isfinite(total):
            continue
        weights /= total
        x = _finite_weighted_mean(weights, exact["diagnostic_decoded_endpoint_x"].to_numpy(dtype=float))
        y = _finite_weighted_mean(weights, exact["diagnostic_decoded_endpoint_y"].to_numpy(dtype=float))
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


def _finite_weighted_mean(weights: np.ndarray, values: np.ndarray) -> float:
    """Return a finite weighted mean without overflowing valid coordinates."""
    coordinate_scale = float(np.max(np.abs(values)))
    if coordinate_scale == 0.0:
        return 0.0

    scaled_values = values / coordinate_scale
    weight_total = math.fsum(float(weight) for weight in weights)
    scaled_mean = math.fsum(
        float(weight) * float(value)
        for weight, value in zip(weights, scaled_values, strict=True)
    ) / weight_total
    scaled_mean = float(np.clip(scaled_mean, np.min(scaled_values), np.max(scaled_values)))
    return float(scaled_mean * coordinate_scale)


def _distinct_model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the strongest finite-evidence row for each normalized model identity."""
    if frame.empty or "model" not in frame.columns:
        return frame
    if "log_evidence" not in frame.columns:
        identities = frame["model"].map(_model_identity)
        return frame.loc[~identities.duplicated(keep="first")].copy()

    evidence = pd.to_numeric(frame["log_evidence"], errors="coerce").to_numpy(dtype=float)
    sort_key = np.where(np.isfinite(evidence), -evidence, np.inf)
    order = np.argsort(sort_key, kind="stable")
    ordered = frame.iloc[order]
    identities = ordered["model"].map(_model_identity)
    return ordered.loc[~identities.duplicated(keep="first")].copy()


def _model_identity(value: object) -> object:
    """Return a hashable model key, decoding byte-backed scalar identifiers."""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _model_identity(value.reshape(-1)[0])
        return ("array", tuple(_model_identity(item) for item in value.reshape(-1)))
    if isinstance(value, np.generic):
        return _model_identity(value.item())
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return ("bytes", raw)
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_model_identity(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


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


def _decoded_text(value: object) -> str | None:
    """Return stripped scalar text, decoding valid byte-backed values."""

    if isinstance(value, np.generic):
        return _decoded_text(value.item())
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
    return str(value).strip()


def _scope_label(value: object) -> str:
    if _is_missing_scalar(value):
        return "<missing>"
    if isinstance(value, (bool, np.bool_)):
        return repr(("scalar", str(bool(value))))
    if isinstance(value, (int, np.integer)):
        return repr(("numeric", int(value)))
    if isinstance(value, (float, np.floating)):
        if bool(np.isfinite(value)):
            numerator, denominator = value.as_integer_ratio()
            if denominator == 1:
                return repr(("numeric", int(numerator)))
            return repr(("numeric-ratio", int(numerator), int(denominator)))
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return _scope_label(array.item())
        value = array.tolist()
    if isinstance(value, (list, tuple)):
        return repr(("sequence", tuple(_scope_label(item) for item in value)))
    if isinstance(value, (set, frozenset)):
        return repr(("set", tuple(sorted(_scope_label(item) for item in value))))
    text = _decoded_text(value)
    if text is not None:
        return repr(("scalar", text))
    raw = value.item() if isinstance(value, np.generic) else value
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return repr(("bytes", bytes(raw)))
    return repr(("scalar", str(raw).strip()))


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
    text = _decoded_text(value)
    return bool(text is not None and text.lower() in _TRUE_BOOL_TEXT_VALUES)


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_bool_value).astype(bool)
