#!/usr/bin/env python3
"""Relate trajectory-family replay dynamics to future behavior.

This post-hoc analysis joins full-core event evidence with behavioral geometry.
For each event it chooses the leading exact trajectory row, extracts decoded
endpoint diagnostics, compares replay direction to previous and future movement
vectors, and summarizes whether trajectory-family confidence or momentum-like
evidence predicts future path alignment.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_model_evidence import _check_session, _session_path
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.ground_truth import active_goal_at_time, infer_well_locations


STATIONARY = "sorted-spike-state-space-stationary"
DIFFUSION = "sorted-spike-state-space-diffusion"
FRAGMENTED = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM_EXACT = "sorted-spike-state-space-momentum-exact-sparse"

REQUIRED_EXACT_CORE_MODELS: tuple[str, ...] = (
    STATIONARY,
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)
TRAJECTORY_MODELS: tuple[str, ...] = (
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)
MODEL_PROBABILITY_COLUMNS = {
    STATIONARY: "posterior_static",
    DIFFUSION: "posterior_diffusion",
    FRAGMENTED: "posterior_fragmented",
    FIRST_ORDER_IMM: "posterior_first_order_imm",
    MOMENTUM_EXACT: "posterior_momentum_exact_sparse",
}

ALIGNMENT_OUTPUT = "replay_behavior_alignment.csv"
PREDICTION_SUMMARY_OUTPUT = "future_path_prediction_summary.csv"
RAT_SUMMARY_OUTPUT = "rat_behavior_alignment_summary.csv"
LOO_OUTPUT = "leave_one_rat_out_behavior_prediction.csv"
_LEGACY_MISSING_TEXT = {"", "nan", "none", "null", "na", "n/a", "<na>"}
_REAL_WINDOW_ROLES = {"real", *_LEGACY_MISSING_TEXT}


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "null", "off"}:
        return False
    try:
        numeric = float(text)
    except ValueError:
        return False
    return bool(np.isfinite(numeric) and numeric != 0.0)


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _parse_names(value: str | Iterable[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
        return names or tuple(default)
    names = tuple(str(part).strip() for part in value if str(part).strip())
    return names or tuple(default)


def _safe_softmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return np.full(arr.shape, np.nan, dtype=float)
    shifted = arr - np.max(arr)
    exp_values = np.exp(shifted)
    total = exp_values.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.full(arr.shape, np.nan, dtype=float)
    return exp_values / total


def _safe_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape or not np.all(np.isfinite(left_arr)) or not np.all(np.isfinite(right_arr)):
        return np.nan
    return float(np.linalg.norm(left_arr - right_arr))


def _safe_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape or not np.all(np.isfinite(left_arr)) or not np.all(np.isfinite(right_arr)):
        return np.nan
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(np.dot(left_arr, right_arr) / denom)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _first_numeric(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _diagnostic_numeric(row: pd.Series, name: str) -> float:
    candidates = (f"diagnostic_{name}", name)
    for column in candidates:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return np.nan


def _normalized_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower()


def _status_success_mask(status: pd.Series) -> pd.Series:
    normalized = _normalized_text(status)
    return normalized.isna() | normalized.eq("success") | normalized.isin(_LEGACY_MISSING_TEXT)


def _real_window_role_mask(window_role: pd.Series) -> pd.Series:
    normalized = _normalized_text(window_role)
    return normalized.isna() | normalized.isin(_REAL_WINDOW_ROLES)


def _success_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event-model evidence table is missing required columns: {missing}")

    out = frame.copy()
    if "status" in out.columns:
        out = out[_status_success_mask(out["status"])].copy()
    if "evidence_comparable" in out.columns:
        out = out[out["evidence_comparable"].map(_as_bool)].copy()
    if "window_role" in out.columns:
        out = out[_real_window_role_mask(out["window_role"])].copy()
    out["session"] = out["session"].astype(str)
    out["rat"] = out["session"].map(_rat_from_session)
    out["event_index"] = pd.to_numeric(out["event_index"], errors="raise").astype(int)
    out["model"] = out["model"].astype(str)
    out["log_evidence"] = pd.to_numeric(out["log_evidence"], errors="coerce")
    return out.dropna(subset=["log_evidence"]).copy()


def build_event_evidence_features(
    event_model_evidence: pd.DataFrame,
    *,
    required_models: Sequence[str] = REQUIRED_EXACT_CORE_MODELS,
    trajectory_models: Sequence[str] = TRAJECTORY_MODELS,
) -> pd.DataFrame:
    """Return one evidence-feature row per complete or attempted replay event."""

    evidence = _success_rows(event_model_evidence)
    required = tuple(str(model) for model in required_models)
    trajectory_set = set(str(model) for model in trajectory_models)
    rows: list[dict[str, object]] = []
    for (session, event_index), group in evidence.groupby(["session", "event_index"], sort=True):
        core = group[group["model"].astype(str).isin(required)].copy()
        by_model = core.drop_duplicates("model", keep="last").set_index("model")
        missing = tuple(model for model in required if model not in by_model.index)
        logz = {
            model: (float(by_model.loc[model, "log_evidence"]) if model in by_model.index else np.nan)
            for model in required
        }
        probabilities = _safe_softmax([logz[model] for model in required]) if not missing else np.full(len(required), np.nan)
        posterior_by_model = dict(zip(required, probabilities, strict=True))
        trajectory = core[core["model"].isin(trajectory_set)].copy()
        nontrajectory = core[~core["model"].isin(trajectory_set)].copy()

        best_trajectory = trajectory.sort_values(["log_evidence", "model"], ascending=[False, True]).iloc[0] if not trajectory.empty else None
        best_nontrajectory = nontrajectory.sort_values(["log_evidence", "model"], ascending=[False, True]).iloc[0] if not nontrajectory.empty else None
        best_trajectory_logz = float(best_trajectory["log_evidence"]) if best_trajectory is not None else np.nan
        best_nontrajectory_logz = float(best_nontrajectory["log_evidence"]) if best_nontrajectory is not None else np.nan
        margin = best_trajectory_logz - best_nontrajectory_logz if np.isfinite(best_trajectory_logz) and np.isfinite(best_nontrajectory_logz) else np.nan

        row = {
            "rat": _rat_from_session(session),
            "session": str(session),
            "event_index": int(event_index),
            "exact_core_complete": bool(not missing),
            "missing_exact_core_models": " ".join(missing),
            "best_trajectory_model": "" if best_trajectory is None else str(best_trajectory["model"]),
            "best_trajectory_log_evidence": best_trajectory_logz,
            "best_nontrajectory_model": "" if best_nontrajectory is None else str(best_nontrajectory["model"]),
            "best_nontrajectory_log_evidence": best_nontrajectory_logz,
            "trajectory_minus_nontrajectory_log_evidence": margin,
            "trajectory_family_confidence": margin,
            "decoded_endpoint_x": _diagnostic_numeric(best_trajectory, "decoded_endpoint_x") if best_trajectory is not None else np.nan,
            "decoded_endpoint_y": _diagnostic_numeric(best_trajectory, "decoded_endpoint_y") if best_trajectory is not None else np.nan,
            "decoded_map_x": _diagnostic_numeric(best_trajectory, "decoded_map_x") if best_trajectory is not None else np.nan,
            "decoded_map_y": _diagnostic_numeric(best_trajectory, "decoded_map_y") if best_trajectory is not None else np.nan,
            "decoded_terminal_entropy": _diagnostic_numeric(best_trajectory, "terminal_posterior_entropy") if best_trajectory is not None else np.nan,
            "mean_trajectory_posterior_entropy": _diagnostic_numeric(best_trajectory, "mean_trajectory_posterior_entropy") if best_trajectory is not None else np.nan,
            "window_start_s": _first_numeric(group, "window_start_s"),
            "window_end_s": _first_numeric(group, "window_end_s"),
            "window_duration_s": _first_numeric(group, "window_duration_s"),
        }
        for model, column in MODEL_PROBABILITY_COLUMNS.items():
            row[column] = float(posterior_by_model.get(model, np.nan))
        row["trajectory_family_posterior_mass"] = float(
            np.nansum([posterior_by_model.get(model, np.nan) for model in trajectory_models])
        ) if not missing else np.nan
        row["momentum_index"] = (
            float(row["posterior_momentum_exact_sparse"] - row["posterior_diffusion"])
            if pd.notna(row["posterior_momentum_exact_sparse"]) and pd.notna(row["posterior_diffusion"])
            else np.nan
        )
        row["first_order_imm_posterior"] = row["posterior_first_order_imm"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_behavior_context_from_dataset(
    event_features: pd.DataFrame,
    dataset_root: str | Path,
    *,
    future_horizon_s: float = 2.0,
    previous_horizon_s: float = 2.0,
) -> pd.DataFrame:
    """Return current/previous/future position and well geometry for evidence events."""

    rows: list[dict[str, object]] = []
    for session_id, group in event_features.groupby("session", sort=True):
        session_dir = _session_path(dataset_root, str(session_id))
        _check_session(session_dir)
        session = load_replay_session(session_dir)
        wells = infer_well_locations(session)
        for event_index in sorted(group["event_index"].astype(int).unique()):
            event = session.ripple(int(event_index))
            anchor_time = float(event.peak)
            current = _position_at_time(session.position, anchor_time)
            previous = _position_at_time(session.position, anchor_time - float(previous_horizon_s))
            future = _position_at_time(session.position, anchor_time + float(future_horizon_s))
            active_goal_id = active_goal_at_time(session, anchor_time)
            active_goal = _well_by_id(wells, active_goal_id)
            nearest_current = _nearest_well(wells, current)
            rows.append(
                {
                    "session": session.session_id,
                    "event_index": int(event_index),
                    "event_start_s": float(event.start),
                    "event_end_s": float(event.end),
                    "event_peak_s": float(event.peak),
                    "current_x": float(current[0]) if np.all(np.isfinite(current)) else np.nan,
                    "current_y": float(current[1]) if np.all(np.isfinite(current)) else np.nan,
                    "previous_x": float(previous[0]) if np.all(np.isfinite(previous)) else np.nan,
                    "previous_y": float(previous[1]) if np.all(np.isfinite(previous)) else np.nan,
                    "future_x": float(future[0]) if np.all(np.isfinite(future)) else np.nan,
                    "future_y": float(future[1]) if np.all(np.isfinite(future)) else np.nan,
                    "future_horizon_s": float(future_horizon_s),
                    "previous_horizon_s": float(previous_horizon_s),
                    "active_goal_id": "" if active_goal_id is None else int(active_goal_id),
                    "active_goal_x": active_goal[0],
                    "active_goal_y": active_goal[1],
                    "nearest_current_well_id": nearest_current["well_id"],
                    "nearest_current_well_x": nearest_current["well_x"],
                    "nearest_current_well_y": nearest_current["well_y"],
                    "well_locations_available": bool(not wells.empty),
                }
            )
    return pd.DataFrame(rows)


def _position_at_time(position: np.ndarray, time_s: float) -> np.ndarray:
    arr = np.asarray(position, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return np.array([np.nan, np.nan], dtype=float)
    keep = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    arr = arr[keep]
    if arr.shape[0] == 0:
        return np.array([np.nan, np.nan], dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    times = arr[:, 0]
    x = np.interp(float(time_s), times, arr[:, 1], left=np.nan, right=np.nan)
    y = np.interp(float(time_s), times, arr[:, 2], left=np.nan, right=np.nan)
    return np.array([float(x), float(y)], dtype=float)


def _well_by_id(wells: pd.DataFrame, well_id: int | None) -> tuple[float, float]:
    if well_id is None or wells.empty:
        return np.nan, np.nan
    row = wells[wells["well_id"].astype(int).eq(int(well_id))]
    if row.empty:
        return np.nan, np.nan
    first = row.iloc[0]
    return float(first["well_x"]), float(first["well_y"])


def _nearest_well(wells: pd.DataFrame, xy: np.ndarray) -> dict[str, object]:
    if wells.empty or not np.all(np.isfinite(xy)):
        return {"well_id": "", "well_x": np.nan, "well_y": np.nan, "distance_cm": np.nan}
    centers = wells[["well_x", "well_y"]].to_numpy(dtype=float)
    distances = np.linalg.norm(centers - np.asarray(xy, dtype=float)[None, :], axis=1)
    idx = int(np.argmin(distances))
    row = wells.iloc[idx]
    return {
        "well_id": int(row["well_id"]),
        "well_x": float(row["well_x"]),
        "well_y": float(row["well_y"]),
        "distance_cm": float(distances[idx]),
    }


def build_replay_behavior_alignment(event_features: pd.DataFrame, behavior_context: pd.DataFrame) -> pd.DataFrame:
    """Join evidence features and behavior context, then compute alignment metrics."""

    if event_features.empty:
        return pd.DataFrame()
    context = behavior_context.copy()
    if context.empty:
        raise ValueError("behavior context is empty")
    out = event_features.merge(context, on=["session", "event_index"], how="left")
    rows: list[dict[str, object]] = []
    for _, row in out.iterrows():
        current = np.array([row.get("current_x", np.nan), row.get("current_y", np.nan)], dtype=float)
        endpoint = np.array([row.get("decoded_endpoint_x", np.nan), row.get("decoded_endpoint_y", np.nan)], dtype=float)
        future = np.array([row.get("future_x", np.nan), row.get("future_y", np.nan)], dtype=float)
        previous = np.array([row.get("previous_x", np.nan), row.get("previous_y", np.nan)], dtype=float)
        active_goal = np.array([row.get("active_goal_x", np.nan), row.get("active_goal_y", np.nan)], dtype=float)
        nearest_well = np.array([row.get("nearest_current_well_x", np.nan), row.get("nearest_current_well_y", np.nan)], dtype=float)

        decoded_direction = endpoint - current
        next_movement = future - current
        previous_path = current - previous
        goal_reference = active_goal if np.all(np.isfinite(active_goal)) else nearest_well
        result = row.to_dict()
        result.update(
            {
                "decoded_direction_x": float(decoded_direction[0]) if np.isfinite(decoded_direction[0]) else np.nan,
                "decoded_direction_y": float(decoded_direction[1]) if np.isfinite(decoded_direction[1]) else np.nan,
                "decoded_direction_norm_cm": float(np.linalg.norm(decoded_direction)) if np.all(np.isfinite(decoded_direction)) else np.nan,
                "next_movement_x": float(next_movement[0]) if np.isfinite(next_movement[0]) else np.nan,
                "next_movement_y": float(next_movement[1]) if np.isfinite(next_movement[1]) else np.nan,
                "next_movement_norm_cm": float(np.linalg.norm(next_movement)) if np.all(np.isfinite(next_movement)) else np.nan,
                "previous_path_x": float(previous_path[0]) if np.isfinite(previous_path[0]) else np.nan,
                "previous_path_y": float(previous_path[1]) if np.isfinite(previous_path[1]) else np.nan,
                "previous_path_norm_cm": float(np.linalg.norm(previous_path)) if np.all(np.isfinite(previous_path)) else np.nan,
                "distance_to_current_position_cm": _safe_distance(endpoint, current),
                "distance_current_to_goal_or_well_cm": _safe_distance(current, goal_reference),
                "distance_endpoint_to_goal_or_well_cm": _safe_distance(endpoint, goal_reference),
                "alignment_with_next_movement": _safe_cosine(decoded_direction, next_movement),
                "alignment_with_previous_path": _safe_cosine(decoded_direction, previous_path),
                "endpoint_closer_to_goal_than_current": (
                    _safe_distance(endpoint, goal_reference) < _safe_distance(current, goal_reference)
                    if np.isfinite(_safe_distance(endpoint, goal_reference)) and np.isfinite(_safe_distance(current, goal_reference))
                    else np.nan
                ),
                "future_path_alignment_positive": (
                    _safe_cosine(decoded_direction, next_movement) > 0.0
                    if np.isfinite(_safe_cosine(decoded_direction, next_movement))
                    else np.nan
                ),
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)


def future_path_prediction_summary(alignment: pd.DataFrame) -> pd.DataFrame:
    """Summarize future movement prediction axes."""

    if alignment.empty:
        return pd.DataFrame()
    valid = alignment.dropna(subset=["alignment_with_next_movement"]).copy()
    rows: list[dict[str, object]] = []

    def add(axis: str, frame: pd.DataFrame, predictor: str = "") -> None:
        y = _numeric(frame, "alignment_with_next_movement").dropna()
        row = {
            "analysis": axis,
            "events": int(len(frame)),
            "valid_alignment_events": int(len(y)),
            "mean_alignment_with_next_movement": float(y.mean()) if not y.empty else np.nan,
            "median_alignment_with_next_movement": float(y.median()) if not y.empty else np.nan,
            "positive_alignment_fraction": float((y > 0.0).mean()) if not y.empty else np.nan,
            "predictor": predictor,
            "predictor_slope": np.nan,
            "predictor_correlation": np.nan,
            "high_minus_low_alignment": np.nan,
        }
        if predictor and predictor in frame.columns:
            slope, corr = _slope_and_correlation(frame[predictor], frame["alignment_with_next_movement"])
            row["predictor_slope"] = slope
            row["predictor_correlation"] = corr
            row["high_minus_low_alignment"] = _high_minus_low(frame, predictor, "alignment_with_next_movement")
        rows.append(row)

    add("endpoint_predicts_next_movement", valid)
    add("trajectory_confidence_predicts_alignment", valid, "trajectory_family_confidence")
    add("trajectory_posterior_predicts_alignment", valid, "trajectory_family_posterior_mass")
    add("momentum_index_predicts_alignment", valid, "momentum_index")
    add("first_order_imm_posterior_predicts_alignment", valid, "first_order_imm_posterior")

    for model, group in valid.groupby("best_trajectory_model", sort=True):
        add(f"best_trajectory_model={model}", group)
    return pd.DataFrame(rows)


def rat_behavior_alignment_summary(alignment: pd.DataFrame) -> pd.DataFrame:
    if alignment.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for rat, group in alignment.groupby("rat", sort=True):
        y = _numeric(group, "alignment_with_next_movement").dropna()
        slope, corr = _slope_and_correlation(group["trajectory_family_confidence"], group["alignment_with_next_movement"])
        rows.append(
            {
                "rat": str(rat),
                "events": int(len(group)),
                "valid_alignment_events": int(len(y)),
                "mean_alignment_with_next_movement": float(y.mean()) if not y.empty else np.nan,
                "median_alignment_with_next_movement": float(y.median()) if not y.empty else np.nan,
                "positive_alignment_fraction": float((y > 0.0).mean()) if not y.empty else np.nan,
                "trajectory_confidence_alignment_slope": slope,
                "trajectory_confidence_alignment_correlation": corr,
                "high_minus_low_trajectory_confidence_alignment": _high_minus_low(
                    group,
                    "trajectory_family_confidence",
                    "alignment_with_next_movement",
                ),
                "mean_momentum_index": float(_numeric(group, "momentum_index").mean()),
                "mean_first_order_imm_posterior": float(_numeric(group, "first_order_imm_posterior").mean()),
            }
        )
    return pd.DataFrame(rows)


def leave_one_rat_out_behavior_prediction(alignment: pd.DataFrame) -> pd.DataFrame:
    if alignment.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    rats = sorted(alignment["rat"].dropna().astype(str).unique())
    for held_out_rat in rats:
        train = alignment[alignment["rat"].astype(str).ne(held_out_rat)].copy()
        test = alignment[alignment["rat"].astype(str).eq(held_out_rat)].copy()
        threshold = float(_numeric(train, "trajectory_family_confidence").median()) if not train.empty else np.nan
        high = test[_numeric(test, "trajectory_family_confidence") >= threshold] if np.isfinite(threshold) else pd.DataFrame()
        low = test[_numeric(test, "trajectory_family_confidence") < threshold] if np.isfinite(threshold) else pd.DataFrame()
        high_alignment = _numeric(high, "alignment_with_next_movement").dropna()
        low_alignment = _numeric(low, "alignment_with_next_movement").dropna()
        all_alignment = _numeric(test, "alignment_with_next_movement").dropna()
        rows.append(
            {
                "held_out_rat": held_out_rat,
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "train_trajectory_confidence_median_threshold": threshold,
                "test_high_confidence_events": int(len(high)),
                "test_low_confidence_events": int(len(low)),
                "test_mean_alignment": float(all_alignment.mean()) if not all_alignment.empty else np.nan,
                "test_high_confidence_mean_alignment": float(high_alignment.mean()) if not high_alignment.empty else np.nan,
                "test_low_confidence_mean_alignment": float(low_alignment.mean()) if not low_alignment.empty else np.nan,
                "test_high_minus_low_alignment": (
                    float(high_alignment.mean() - low_alignment.mean())
                    if not high_alignment.empty and not low_alignment.empty
                    else np.nan
                ),
                "test_positive_alignment_fraction": float((all_alignment > 0.0).mean()) if not all_alignment.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _slope_and_correlation(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2:
        return np.nan, np.nan
    x_values = frame["x"].to_numpy(dtype=float)
    y_values = frame["y"].to_numpy(dtype=float)
    slope = float(np.polyfit(x_values, y_values, deg=1)[0])
    corr = float(np.corrcoef(x_values, y_values)[0, 1]) if np.std(x_values) > 0 and np.std(y_values) > 0 else np.nan
    return slope, corr


def _high_minus_low(frame: pd.DataFrame, predictor: str, outcome: str) -> float:
    working = frame[[predictor, outcome]].copy()
    working[predictor] = pd.to_numeric(working[predictor], errors="coerce")
    working[outcome] = pd.to_numeric(working[outcome], errors="coerce")
    working = working.dropna()
    if len(working) < 2 or working[predictor].nunique() < 2:
        return np.nan
    threshold = float(working[predictor].median())
    high = working[working[predictor] >= threshold][outcome]
    low = working[working[predictor] < threshold][outcome]
    if high.empty or low.empty:
        return np.nan
    return float(high.mean() - low.mean())


def write_replay_behavior_alignment_outputs(
    event_model_evidence: pd.DataFrame,
    output: str | Path,
    *,
    behavior_context: pd.DataFrame,
    required_models: Sequence[str] = REQUIRED_EXACT_CORE_MODELS,
) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    features = build_event_evidence_features(event_model_evidence, required_models=required_models)
    alignment = build_replay_behavior_alignment(features, behavior_context)
    outputs = {
        ALIGNMENT_OUTPUT: alignment,
        PREDICTION_SUMMARY_OUTPUT: future_path_prediction_summary(alignment),
        RAT_SUMMARY_OUTPUT: rat_behavior_alignment_summary(alignment),
        LOO_OUTPUT: leave_one_rat_out_behavior_prediction(alignment),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-model-evidence", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--behavior-context", default="")
    parser.add_argument("--output", default="results/replay-behavior-alignment")
    parser.add_argument("--required-models", default=" ".join(REQUIRED_EXACT_CORE_MODELS))
    parser.add_argument("--future-horizon-s", type=float, default=2.0)
    parser.add_argument("--previous-horizon-s", type=float, default=2.0)
    args = parser.parse_args()

    evidence = pd.read_csv(args.event_model_evidence)
    required_models = _parse_names(args.required_models, REQUIRED_EXACT_CORE_MODELS)
    features = build_event_evidence_features(evidence, required_models=required_models)
    contexts: list[pd.DataFrame] = []
    if str(args.behavior_context).strip():
        contexts.append(pd.read_csv(args.behavior_context))
    if str(args.dataset_root).strip():
        contexts.append(
            build_behavior_context_from_dataset(
                features,
                args.dataset_root,
                future_horizon_s=args.future_horizon_s,
                previous_horizon_s=args.previous_horizon_s,
            )
        )
    if not contexts:
        raise ValueError("provide --dataset-root or --behavior-context to compute behavioral alignment")
    behavior_context = pd.concat(contexts, ignore_index=True).drop_duplicates(["session", "event_index"], keep="first")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    alignment = build_replay_behavior_alignment(features, behavior_context)
    outputs = {
        ALIGNMENT_OUTPUT: alignment,
        PREDICTION_SUMMARY_OUTPUT: future_path_prediction_summary(alignment),
        RAT_SUMMARY_OUTPUT: rat_behavior_alignment_summary(alignment),
        LOO_OUTPUT: leave_one_rat_out_behavior_prediction(alignment),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)

    print("Replay behavior alignment summary:")
    print(outputs[PREDICTION_SUMMARY_OUTPUT].to_string(index=False))
    print("\nRat behavior alignment summary:")
    print(outputs[RAT_SUMMARY_OUTPUT].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
