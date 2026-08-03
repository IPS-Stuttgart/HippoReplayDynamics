#!/usr/bin/env python3
"""Freeze and audit feasibility for the replay commitment/composition test.

This is deliberately a non-outcome audit. Model classifications and behavioral
coverage are written to separate tables, and no relationship between them is
estimated. The script predeclares whether the categorical momentum subset is
large enough or whether the continuous exact-momentum-minus-IMM margin must be
the primary predictor before behavioral outcomes are inspected.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
from hipporeplayimm.data import load_replay_session  # noqa: E402


STATIONARY = "sorted-spike-state-space-stationary"
DIFFUSION = "sorted-spike-state-space-diffusion"
FRAGMENTED = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM = "sorted-spike-state-space-momentum-exact-sparse"

EXACT_CORE_MODELS: tuple[str, ...] = (
    STATIONARY,
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM,
)
LOGZ_COLUMNS = {
    STATIONARY: "logZ_stationary",
    DIFFUSION: "logZ_diffusion",
    FRAGMENTED: "logZ_fragmented",
    FIRST_ORDER_IMM: "logZ_first_order_imm",
    MOMENTUM: "logZ_momentum_exact_sparse",
}

EVENT_TABLE_OUTPUT = "replay_commitment_composition_frozen_events.csv"
MODEL_COUNT_OUTPUT = "replay_commitment_composition_model_counts.csv"
PRIMARY_AXIS_OUTPUT = "replay_commitment_composition_primary_axis.csv"
BEHAVIOR_COVERAGE_OUTPUT = "replay_commitment_composition_behavior_coverage.csv"
BEHAVIOR_SUMMARY_OUTPUT = "replay_commitment_composition_behavior_coverage_summary.csv"
EXTERNAL_OUTPUT = "replay_commitment_composition_external_dataset_suitability.csv"
GATE_OUTPUT = "replay_commitment_composition_feasibility_gate_summary.csv"
MANIFEST_OUTPUT = "replay_commitment_composition_feasibility_manifest.json"
SUMMARY_OUTPUT = "replay_commitment_composition_feasibility_summary.md"

_MISSING_TEXT = {"", "nan", "none", "null", "na", "n/a", "<na>"}


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
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return bool(np.isfinite(number) and number != 0.0)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "on"}:
        return True
    if text in {"false", "f", "no", "n", "off", *_MISSING_TEXT}:
        return False
    try:
        number = float(text)
    except ValueError:
        return False
    return bool(np.isfinite(number) and number != 0.0)


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _successful_exact_rows(evidence: pd.DataFrame) -> pd.DataFrame:
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(evidence.columns))
    if missing:
        raise ValueError(f"event-model evidence is missing required columns: {missing}")
    out = evidence.copy()
    if "status" in out:
        status = out["status"].astype("string").str.strip().str.lower()
        out = out[status.isna() | status.eq("success") | status.isin(_MISSING_TEXT)].copy()
    if "evidence_comparable" in out:
        out = out[out["evidence_comparable"].map(_as_bool)].copy()
    if "window_role" in out:
        role = out["window_role"].astype("string").str.strip().str.lower()
        out = out[role.isna() | role.eq("real") | role.isin(_MISSING_TEXT)].copy()
    out["session"] = out["session"].astype(str)
    out["event_index"] = pd.to_numeric(out["event_index"], errors="raise").astype(int)
    out["model"] = out["model"].astype(str)
    out["log_evidence"] = pd.to_numeric(out["log_evidence"], errors="coerce")
    return out[out["model"].isin(EXACT_CORE_MODELS)].dropna(subset=["log_evidence"]).copy()


def build_frozen_event_table(
    evidence: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    """Return one frozen model-only row per event."""

    exact = _successful_exact_rows(evidence)
    rows: list[dict[str, object]] = []
    for (session, event_index), group in exact.groupby(["session", "event_index"], sort=True):
        by_model = group.drop_duplicates("model", keep="last").set_index("model")
        missing_models = [model for model in EXACT_CORE_MODELS if model not in by_model.index]
        logz = {
            model: float(by_model.loc[model, "log_evidence"]) if model in by_model.index else np.nan
            for model in EXACT_CORE_MODELS
        }
        complete = not missing_models and all(np.isfinite(value) for value in logz.values())
        ordered = sorted(
            ((model, value) for model, value in logz.items() if np.isfinite(value)),
            key=lambda item: (-item[1], item[0]),
        )
        best_model, best_logz = ordered[0] if ordered else ("", np.nan)
        runner_up, runner_up_logz = ordered[1] if len(ordered) > 1 else ("", np.nan)
        best_margin = best_logz - runner_up_logz if np.isfinite(best_logz) and np.isfinite(runner_up_logz) else np.nan
        delta_imm_fragmented = logz[FIRST_ORDER_IMM] - logz[FRAGMENTED]
        delta_momentum_imm = logz[MOMENTUM] - logz[FIRST_ORDER_IMM]
        best_other_momentum = max(value for model, value in logz.items() if model != MOMENTUM)
        delta_momentum_other = logz[MOMENTUM] - best_other_momentum
        confident_best = bool(complete and best_margin >= float(margin_threshold))
        clean_imm = bool(
            complete
            and best_model == FIRST_ORDER_IMM
            and delta_imm_fragmented >= float(margin_threshold)
        )
        momentum_like = bool(complete and best_model == MOMENTUM and confident_best)
        if clean_imm:
            role = "clean_imm"
        elif momentum_like:
            role = "momentum_like"
        elif confident_best and best_model == FRAGMENTED:
            role = "fragmented_control"
        elif confident_best and best_model == STATIONARY:
            role = "stationary_control"
        elif confident_best and best_model == DIFFUSION:
            role = "diffusion_control"
        else:
            role = "ambiguous"
        row: dict[str, object] = {
            "session": str(session),
            "rat": _rat_from_session(session),
            "event_index": int(event_index),
            "exact_core_complete": bool(complete),
            "missing_exact_core_models": " ".join(missing_models),
            "margin_threshold": float(margin_threshold),
            "best_exact_core_model": best_model,
            "runner_up_exact_core_model": runner_up,
            "best_minus_runner_up_log_evidence": best_margin,
            "delta_imm_minus_fragmented": delta_imm_fragmented,
            "delta_momentum_minus_imm": delta_momentum_imm,
            "delta_momentum_minus_best_other": delta_momentum_other,
            "clean_imm": clean_imm,
            "raw_momentum_win": bool(complete and best_model == MOMENTUM),
            "confident_momentum_win": momentum_like,
            "confident_exact_core_claim": confident_best,
            "analysis_role": role,
            "n_time": _first_numeric(group, "n_time"),
            "n_spikes": _first_numeric(group, "n_spikes"),
            "frozen_before_behavior": True,
        }
        row.update({column: logz[model] for model, column in LOGZ_COLUMNS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _first_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def build_model_count_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", events)]
    groups.extend(("rat", str(name), group) for name, group in events.groupby("rat", sort=True))
    groups.extend(("session", str(name), group) for name, group in events.groupby("session", sort=True))
    for level, group_id, group in groups:
        rows.append(
            {
                "level": level,
                "group": group_id,
                "events": int(len(group)),
                "exact_core_complete_events": int(group["exact_core_complete"].sum()),
                "clean_imm_events": int(group["clean_imm"].sum()),
                "confident_momentum_events": int(group["confident_momentum_win"].sum()),
                "raw_momentum_wins": int(group["raw_momentum_win"].sum()),
                "ambiguous_events": int(group["analysis_role"].eq("ambiguous").sum()),
                "fragmented_control_events": int(group["analysis_role"].eq("fragmented_control").sum()),
                "stationary_control_events": int(group["analysis_role"].eq("stationary_control").sum()),
                "diffusion_control_events": int(group["analysis_role"].eq("diffusion_control").sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_primary_momentum_axis(
    events: pd.DataFrame,
    *,
    minimum_confident_events: int = 10,
    minimum_confident_rats: int = 3,
) -> pd.DataFrame:
    confident = events[events["confident_momentum_win"]].copy()
    event_count = int(len(confident))
    rat_count = int(confident["rat"].nunique()) if not confident.empty else 0
    categorical_ready = bool(
        event_count >= int(minimum_confident_events)
        and rat_count >= int(minimum_confident_rats)
    )
    return pd.DataFrame(
        [
            {
                "confident_momentum_events": event_count,
                "confident_momentum_rats": rat_count,
                "minimum_confident_momentum_events": int(minimum_confident_events),
                "minimum_confident_momentum_rats": int(minimum_confident_rats),
                "categorical_momentum_primary_ready": categorical_ready,
                "primary_predictor": (
                    "confident_momentum_class"
                    if categorical_ready
                    else "delta_momentum_minus_imm"
                ),
                "categorical_model_classes_role": (
                    "primary"
                    if categorical_ready
                    else "secondary_descriptive_only"
                ),
                "decision_frozen_before_behavior": True,
            }
        ]
    )


def _finite_position(position: np.ndarray) -> np.ndarray:
    array = np.asarray(position, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        return np.empty((0, 4), dtype=float)
    keep = np.isfinite(array[:, 0]) & np.isfinite(array[:, 1]) & np.isfinite(array[:, 2])
    array = array[keep]
    if array.shape[0] == 0:
        return np.empty((0, 4), dtype=float)
    order = np.argsort(array[:, 0], kind="stable")
    array = array[order]
    if array.shape[1] >= 4:
        speed = array[:, 3]
    else:
        dt = np.diff(array[:, 0])
        distance = np.linalg.norm(np.diff(array[:, 1:3], axis=0), axis=1)
        speed = np.r_[distance / np.where(dt > 0.0, dt, np.nan), np.nan]
    return np.column_stack([array[:, :3], speed])


def _inside_intervals(time_s: float, intervals: np.ndarray) -> bool:
    array = np.asarray(intervals, dtype=float)
    if array.ndim == 1 and array.size == 2:
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        return False
    return bool(np.any((time_s >= array[:, 0]) & (time_s <= array[:, 1])))


def _path_metrics(position: np.ndarray, start_s: float, end_s: float) -> tuple[int, float, float]:
    if position.shape[0] == 0 or not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return 0, np.nan, np.nan
    segment = position[(position[:, 0] >= start_s) & (position[:, 0] <= end_s)]
    if len(segment) < 2:
        return int(len(segment)), np.nan, np.nan
    steps = np.linalg.norm(np.diff(segment[:, 1:3], axis=0), axis=1)
    path_length = float(np.nansum(steps))
    displacement = float(np.linalg.norm(segment[-1, 1:3] - segment[0, 1:3]))
    return int(len(segment)), path_length, displacement


def _well_context(well_sequence: np.ndarray, peak_s: float) -> dict[str, object]:
    array = np.asarray(well_sequence, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        return {
            "previous_well_time_s": np.nan,
            "previous_well_id": np.nan,
            "next_well_time_s": np.nan,
            "next_well_id": np.nan,
        }
    finite = np.isfinite(array[:, 0]) & np.isfinite(array[:, 1])
    array = array[finite]
    previous = array[array[:, 0] <= peak_s]
    following = array[array[:, 0] > peak_s]
    return {
        "previous_well_time_s": float(previous[-1, 0]) if len(previous) else np.nan,
        "previous_well_id": float(previous[-1, 1]) if len(previous) else np.nan,
        "next_well_time_s": float(following[0, 0]) if len(following) else np.nan,
        "next_well_id": float(following[0, 1]) if len(following) else np.nan,
    }


def _first_sustained_departure(
    position: np.ndarray,
    peak_s: float,
    *,
    speed_threshold_cm_s: float,
    minimum_sustained_s: float,
    maximum_wait_s: float,
) -> float:
    segment = position[
        (position[:, 0] >= peak_s)
        & (position[:, 0] <= peak_s + maximum_wait_s)
    ]
    if len(segment) < 2:
        return np.nan
    dt = float(np.nanmedian(np.diff(segment[:, 0])))
    if not np.isfinite(dt) or dt <= 0.0:
        return np.nan
    required = max(1, int(np.ceil(minimum_sustained_s / dt)))
    moving = np.isfinite(segment[:, 3]) & (segment[:, 3] >= speed_threshold_cm_s)
    run = 0
    for index, value in enumerate(moving):
        run = run + 1 if value else 0
        if run >= required:
            return float(segment[index - required + 1, 0])
    return np.nan


def build_behavior_coverage(
    events: pd.DataFrame,
    dataset_root: str | Path,
    *,
    horizons_s: Sequence[float] = (2.0, 5.0, 10.0, 30.0, 60.0),
    speed_threshold_cm_s: float = 5.0,
    minimum_sustained_departure_s: float = 0.25,
    maximum_departure_wait_s: float = 60.0,
    minimum_route_path_length_cm: float = 10.0,
    session_loader: Callable[[Path], object] = load_replay_session,
) -> pd.DataFrame:
    """Return behavior availability only, without model-class columns."""

    dataset = Path(dataset_root)
    rows: list[dict[str, object]] = []
    for session_id, group in events.groupby("session", sort=True):
        session = session_loader(dataset / Path(str(session_id)))
        position = _finite_position(np.asarray(session.position))
        for event_index in sorted(group["event_index"].astype(int).unique()):
            ripple = session.ripple(int(event_index))
            peak = float(ripple.peak)
            wells = _well_context(np.asarray(session.well_sequence), peak)
            departure = _first_sustained_departure(
                position,
                peak,
                speed_threshold_cm_s=float(speed_threshold_cm_s),
                minimum_sustained_s=float(minimum_sustained_departure_s),
                maximum_wait_s=float(maximum_departure_wait_s),
            )
            previous_well_time = float(wells["previous_well_time_s"])
            next_well_time = float(wells["next_well_time_s"])
            past_samples, past_length, past_displacement = _path_metrics(position, previous_well_time, peak)
            future_start = departure if np.isfinite(departure) else peak
            future_samples, future_length, future_displacement = _path_metrics(position, future_start, next_well_time)
            row: dict[str, object] = {
                "session": str(session_id),
                "rat": _rat_from_session(session_id),
                "event_index": int(event_index),
                "event_start_s": float(ripple.start),
                "event_end_s": float(ripple.end),
                "event_peak_s": peak,
                "event_in_run_epoch": _inside_intervals(peak, np.asarray(session.run_times)),
                "position_at_event_available": bool(
                    len(position)
                    and peak >= float(position[0, 0])
                    and peak <= float(position[-1, 0])
                ),
                **wells,
                "time_since_previous_well_s": peak - previous_well_time if np.isfinite(previous_well_time) else np.nan,
                "time_to_next_well_s": next_well_time - peak if np.isfinite(next_well_time) else np.nan,
                "departure_time_s": departure,
                "time_to_departure_s": departure - peak if np.isfinite(departure) else np.nan,
                "past_route_position_samples": past_samples,
                "past_route_path_length_cm": past_length,
                "past_route_displacement_cm": past_displacement,
                "future_route_position_samples": future_samples,
                "future_route_path_length_cm": future_length,
                "future_route_displacement_cm": future_displacement,
                "past_route_available": bool(
                    past_samples >= 2
                    and np.isfinite(previous_well_time)
                ),
                "future_route_available": bool(
                    np.isfinite(departure)
                    and np.isfinite(next_well_time)
                    and future_samples >= 2
                ),
                "past_route_informative_10cm": bool(
                    past_samples >= 2
                    and np.isfinite(past_length)
                    and past_length >= float(minimum_route_path_length_cm)
                ),
                "future_route_informative_10cm": bool(
                    np.isfinite(departure)
                    and future_samples >= 2
                    and np.isfinite(future_length)
                    and future_length >= float(minimum_route_path_length_cm)
                ),
            }
            for horizon in horizons_s:
                label = f"{float(horizon):g}s"
                past_count, past_horizon_length, _ = _path_metrics(position, peak - float(horizon), peak)
                future_count, future_horizon_length, _ = _path_metrics(position, peak, peak + float(horizon))
                row[f"past_position_samples_{label}"] = past_count
                row[f"future_position_samples_{label}"] = future_count
                row[f"past_path_length_cm_{label}"] = past_horizon_length
                row[f"future_path_length_cm_{label}"] = future_horizon_length
            rows.append(row)
    return pd.DataFrame(rows)


def build_behavior_coverage_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "level",
        "group",
        "events",
        "events_in_run_epoch",
        "position_at_event_available",
        "past_route_available",
        "future_route_available",
        "future_route_available_fraction",
        "past_route_informative_10cm",
        "future_route_informative_10cm",
        "future_route_informative_10cm_fraction",
    ]
    if coverage.empty:
        return pd.DataFrame(columns=columns)
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", coverage)]
    groups.extend(("rat", str(name), group) for name, group in coverage.groupby("rat", sort=True))
    groups.extend(("session", str(name), group) for name, group in coverage.groupby("session", sort=True))
    rows = []
    for level, group_id, group in groups:
        rows.append(
            {
                "level": level,
                "group": group_id,
                "events": int(len(group)),
                "events_in_run_epoch": int(group["event_in_run_epoch"].sum()),
                "position_at_event_available": int(group["position_at_event_available"].sum()),
                "past_route_available": int(group["past_route_available"].sum()),
                "future_route_available": int(group["future_route_available"].sum()),
                "future_route_available_fraction": float(group["future_route_available"].mean()),
                "past_route_informative_10cm": int(group["past_route_informative_10cm"].sum()),
                "future_route_informative_10cm": int(group["future_route_informative_10cm"].sum()),
                "future_route_informative_10cm_fraction": float(group["future_route_informative_10cm"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _load_hc11_position(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    data = loadmat(path, simplify_cells=True)["position"]
    timestamps = np.asarray(data.get("timestamps", []), dtype=float).ravel()
    xy = data.get("position", {})
    x = np.asarray(xy.get("x", []), dtype=float).ravel()
    y = np.asarray(xy.get("y", []), dtype=float).ravel()
    maze = np.asarray(data.get("Epochs", {}).get("MazeEpoch", []), dtype=float).ravel()
    units = str(data.get("units", ""))
    scale = 100.0 if "meter" in units.lower() or units.strip().lower() == "m" else 1.0
    keep = np.isfinite(timestamps) & np.isfinite(x) & np.isfinite(y)
    position = np.column_stack([timestamps[keep], x[keep] * scale, y[keep] * scale])
    return position, maze, str(data.get("behaviorinfo", {}).get("MazeType", ""))


def _load_hc11_ripple_peaks(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["ripples/peaks"], dtype=float).ravel()


def audit_hc11_awake_suitability(
    dataset_root: str | Path,
    *,
    minimum_animals: int = 3,
    minimum_sessions: int = 3,
    future_horizon_s: float = 2.0,
    speed_threshold_cm_s: float = 5.0,
    maximum_departure_wait_s: float = 60.0,
    minimum_future_path_length_cm: float = 10.0,
) -> pd.DataFrame:
    """Count published all-state hc-11 ripples with maze-position coverage."""

    root = Path(dataset_root)
    rows: list[dict[str, object]] = []
    for session_dir in sorted(path for path in root.glob("*/*") if path.is_dir()):
        session = session_dir.name
        position_path = session_dir / f"{session}.position.behavior.mat"
        ripple_path = session_dir / f"{session}.ripplesALL.event.mat"
        if not position_path.exists() or not ripple_path.exists():
            continue
        try:
            position, maze, maze_type = _load_hc11_position(position_path)
            peaks = _load_hc11_ripple_peaks(ripple_path)
        except Exception:
            continue
        position = _finite_position(position)
        if len(position) == 0 or maze.size < 2:
            continue
        in_maze = (peaks >= maze[0]) & (peaks <= maze[1])
        position_covered = in_maze & (peaks >= position[0, 0]) & (peaks <= position[-1, 0])
        future_covered = np.zeros(len(peaks), dtype=bool)
        departure_available = np.zeros(len(peaks), dtype=bool)
        for index in np.flatnonzero(position_covered):
            peak = float(peaks[index])
            departure = _first_sustained_departure(
                position,
                peak,
                speed_threshold_cm_s=float(speed_threshold_cm_s),
                minimum_sustained_s=0.25,
                maximum_wait_s=float(maximum_departure_wait_s),
            )
            if not np.isfinite(departure):
                continue
            departure_available[index] = True
            _, path_length, _ = _path_metrics(
                position,
                departure,
                departure + float(future_horizon_s),
            )
            future_covered[index] = bool(
                np.isfinite(path_length)
                and path_length >= float(minimum_future_path_length_cm)
            )
        rows.append(
            {
                "dataset": "hc11",
                "animal": session_dir.parent.name,
                "session": session,
                "environment": maze_type,
                "event_definition": "published_ripplesALL_intersect_MAZE",
                "published_all_state_ripples_available": True,
                "awake_maze_events": int(in_maze.sum()),
                "events_with_position_at_event": int(position_covered.sum()),
                "events_with_departure_within_window": int(departure_available.sum()),
                "events_with_future_position_coverage": int(future_covered.sum()),
            }
        )
    session_frame = pd.DataFrame(rows)
    if session_frame.empty:
        return pd.DataFrame(
            [
                {
                    "dataset": "hc11",
                    "status": "not_available",
                    "animals": 0,
                    "sessions": 0,
                    "awake_maze_events": 0,
                    "events_with_departure_within_window": 0,
                    "events_with_future_position_coverage": 0,
                    "suitable_for_commitment_confirmation": False,
                    "reason": "no published ripplesALL sessions with MAZE position were found",
                }
            ]
        )
    animals = int(session_frame["animal"].nunique())
    sessions = int(session_frame["session"].nunique())
    events = int(session_frame["awake_maze_events"].sum())
    departures = int(session_frame["events_with_departure_within_window"].sum())
    future = int(session_frame["events_with_future_position_coverage"].sum())
    suitable = bool(
        animals >= int(minimum_animals)
        and sessions >= int(minimum_sessions)
        and events > 0
        and future > 0
    )
    summary = {
        "dataset": "hc11",
        "status": "available" if events > 0 else "no_awake_events",
        "animals": animals,
        "sessions": sessions,
        "awake_maze_events": events,
        "events_with_departure_within_window": departures,
        "events_with_future_position_coverage": future,
        "suitable_for_commitment_confirmation": suitable,
        "reason": (
            "published all-state maze ripples span the predeclared animal/session minimum"
            if suitable
            else "awake events exist, but current published all-state coverage is below the confirmation animal/session minimum"
        ),
    }
    return pd.concat([pd.DataFrame([summary]), session_frame], ignore_index=True, sort=False)


def build_gates(
    events: pd.DataFrame,
    primary_axis: pd.DataFrame,
    coverage: pd.DataFrame,
    external: pd.DataFrame,
    *,
    minimum_behavior_coverage_fraction: float,
    minimum_informative_future_routes: int = 100,
) -> pd.DataFrame:
    event_count = int(len(events))
    complete = int(events["exact_core_complete"].sum()) if not events.empty else 0
    coverage_count = int(len(coverage))
    run_fraction = float(coverage["event_in_run_epoch"].mean()) if coverage_count else 0.0
    future_fraction = float(coverage["future_route_available"].mean()) if coverage_count else 0.0
    informative_future = int(coverage["future_route_informative_10cm"].sum()) if coverage_count else 0
    informative_rats = int(
        coverage.loc[coverage["future_route_informative_10cm"], "rat"].nunique()
    ) if coverage_count else 0
    external_summary = external[external["status"].notna()] if "status" in external else pd.DataFrame()
    external_ready = bool(
        not external_summary.empty
        and external_summary["suitable_for_commitment_confirmation"].map(_as_bool).any()
    )
    primary = str(primary_axis.iloc[0]["primary_predictor"]) if not primary_axis.empty else ""
    checks = [
        ("selected_events_present", event_count > 0, f"events={event_count}"),
        ("exact_core_complete", event_count > 0 and complete == event_count, f"complete={complete}/{event_count}"),
        ("all_four_pf_rats_represented", events["rat"].nunique() == 4 if event_count else False, f"rats={events['rat'].nunique() if event_count else 0}/4"),
        ("primary_axis_frozen_before_behavior", primary in {"confident_momentum_class", "delta_momentum_minus_imm"}, f"primary_predictor={primary}"),
        ("behavior_coverage_present", coverage_count == event_count and event_count > 0, f"coverage={coverage_count}/{event_count}"),
        ("events_in_run_epoch", run_fraction >= minimum_behavior_coverage_fraction, f"fraction={run_fraction:.6g}"),
        ("future_route_coverage", future_fraction >= minimum_behavior_coverage_fraction, f"fraction={future_fraction:.6g}"),
        (
            "informative_future_route_cohort",
            informative_future >= int(minimum_informative_future_routes) and informative_rats == 4,
            f"informative_events={informative_future}/{minimum_informative_future_routes}; rats={informative_rats}/4",
        ),
        ("outcome_join_not_performed", True, "model and behavior tables remain separate"),
        ("external_confirmation_ready", external_ready, f"ready={external_ready}"),
    ]
    pf_ready = all(value for name, value, _ in checks if name not in {"external_confirmation_ready"})
    checks.append(("pf_primary_analysis_ready", pf_ready, "all PF Phase 0 gates required"))
    checks.append(("strong_claim_replication_ready", pf_ready and external_ready, "PF readiness plus independent awake-event coverage required"))
    return pd.DataFrame(checks, columns=["gate", "passed", "detail"])


def _summary_markdown(
    events: pd.DataFrame,
    primary_axis: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    external: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    overall_counts = build_model_count_summary(events).query("level == 'overall'").iloc[0]
    axis = primary_axis.iloc[0]
    coverage = coverage_summary.query("level == 'overall'")
    coverage_row = coverage.iloc[0] if not coverage.empty else None
    external_summary = external[external["status"].notna()] if "status" in external else pd.DataFrame()
    lines = [
        "# Replay commitment/composition feasibility audit",
        "",
        "## Frozen model decision",
        "",
        f"- Exact-core events: {int(overall_counts['exact_core_complete_events'])}/{int(overall_counts['events'])}",
        f"- Clean IMM events: {int(overall_counts['clean_imm_events'])}",
        f"- Confident momentum events: {int(axis['confident_momentum_events'])} across {int(axis['confident_momentum_rats'])} rats",
        f"- Primary predictor: `{axis['primary_predictor']}`",
        f"- Categorical model classes: `{axis['categorical_model_classes_role']}`",
        "",
        "This decision was made from event counts only, before any model-by-behavior outcome was computed.",
        "Model classifications and behavioral coverage remain in separate output tables.",
        "",
        "## Behavioral availability",
        "",
    ]
    if coverage_row is None:
        lines.append("Behavioral coverage was not provided; PF readiness therefore fails non-vacuously.")
    else:
        lines.extend(
            [
                f"- Events in RUN: {int(coverage_row['events_in_run_epoch'])}/{int(coverage_row['events'])}",
                f"- Events with past-route coverage: {int(coverage_row['past_route_available'])}/{int(coverage_row['events'])}",
                f"- Events with future-route coverage: {int(coverage_row['future_route_available'])}/{int(coverage_row['events'])}",
                f"- Events with at least 10 cm of future route: {int(coverage_row['future_route_informative_10cm'])}/{int(coverage_row['events'])}",
            ]
        )
    lines.extend(["", "## External confirmation", ""])
    if external_summary.empty:
        lines.append("No external awake-event dataset was audited.")
    else:
        for _, row in external_summary.iterrows():
            lines.append(
                f"- {row['dataset']}: {int(row.get('awake_maze_events', 0))} awake maze events, "
                f"{int(row.get('animals', 0))} animals, suitable={bool(row.get('suitable_for_commitment_confirmation', False))}"
            )
    lines.extend(["", "## Gates", ""])
    for _, row in gates.iterrows():
        lines.append(f"- {'PASS' if bool(row['passed']) else 'FAIL'} `{row['gate']}`: {row['detail']}")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This artifact establishes feasibility and freezes the predictor choice only. It does not test whether momentum predicts commitment or whether IMM predicts composition.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    evidence: pd.DataFrame,
    output_dir: str | Path,
    *,
    dataset_root: str | Path | None,
    hc11_dataset_root: str | Path | None,
    margin_threshold: float,
    minimum_confident_momentum_events: int,
    minimum_confident_momentum_rats: int,
    minimum_behavior_coverage_fraction: float,
    minimum_informative_future_routes: int,
    minimum_external_animals: int,
    minimum_external_sessions: int,
) -> dict[str, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = build_frozen_event_table(evidence, margin_threshold=margin_threshold)
    model_counts = build_model_count_summary(events)
    primary_axis = choose_primary_momentum_axis(
        events,
        minimum_confident_events=minimum_confident_momentum_events,
        minimum_confident_rats=minimum_confident_momentum_rats,
    )
    coverage = (
        build_behavior_coverage(events, dataset_root)
        if dataset_root not in {None, ""}
        else pd.DataFrame()
    )
    coverage_summary = build_behavior_coverage_summary(coverage)
    external = (
        audit_hc11_awake_suitability(
            hc11_dataset_root,
            minimum_animals=minimum_external_animals,
            minimum_sessions=minimum_external_sessions,
        )
        if hc11_dataset_root not in {None, ""}
        else pd.DataFrame(
            [
                {
                    "dataset": "not_provided",
                    "status": "not_provided",
                    "animals": 0,
                    "sessions": 0,
                    "awake_maze_events": 0,
                    "events_with_future_position_coverage": 0,
                    "suitable_for_commitment_confirmation": False,
                    "reason": "no external dataset root supplied",
                }
            ]
        )
    )
    gates = build_gates(
        events,
        primary_axis,
        coverage,
        external,
        minimum_behavior_coverage_fraction=minimum_behavior_coverage_fraction,
        minimum_informative_future_routes=minimum_informative_future_routes,
    )
    outputs = {
        EVENT_TABLE_OUTPUT: events,
        MODEL_COUNT_OUTPUT: model_counts,
        PRIMARY_AXIS_OUTPUT: primary_axis,
        BEHAVIOR_COVERAGE_OUTPUT: coverage,
        BEHAVIOR_SUMMARY_OUTPUT: coverage_summary,
        EXTERNAL_OUTPUT: external,
        GATE_OUTPUT: gates,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    (output / SUMMARY_OUTPUT).write_text(
        _summary_markdown(events, primary_axis, coverage_summary, external, gates),
        encoding="utf-8",
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-model-evidence", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--hc11-dataset-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--minimum-confident-momentum-events", type=int, default=10)
    parser.add_argument("--minimum-confident-momentum-rats", type=int, default=3)
    parser.add_argument("--minimum-behavior-coverage-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-informative-future-routes", type=int, default=100)
    parser.add_argument("--minimum-external-animals", type=int, default=3)
    parser.add_argument("--minimum-external-sessions", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evidence_path = Path(args.event_model_evidence)
    evidence = pd.read_csv(evidence_path)
    outputs = write_outputs(
        evidence,
        args.output_dir,
        dataset_root=args.dataset_root,
        hc11_dataset_root=args.hc11_dataset_root,
        margin_threshold=args.margin_threshold,
        minimum_confident_momentum_events=args.minimum_confident_momentum_events,
        minimum_confident_momentum_rats=args.minimum_confident_momentum_rats,
        minimum_behavior_coverage_fraction=args.minimum_behavior_coverage_fraction,
        minimum_informative_future_routes=args.minimum_informative_future_routes,
        minimum_external_animals=args.minimum_external_animals,
        minimum_external_sessions=args.minimum_external_sessions,
    )
    provenance = build_script_provenance(
        input_paths={
            "event_model_evidence": evidence_path,
            "dataset_root": args.dataset_root,
            "hc11_dataset_root": args.hc11_dataset_root,
        }
    )
    provenance.update(
        {
            "analysis": "replay_commitment_composition_feasibility_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "margin_threshold": float(args.margin_threshold),
            "minimum_confident_momentum_events": int(args.minimum_confident_momentum_events),
            "minimum_confident_momentum_rats": int(args.minimum_confident_momentum_rats),
            "minimum_informative_future_routes": int(args.minimum_informative_future_routes),
            "primary_predictor": str(outputs[PRIMARY_AXIS_OUTPUT].iloc[0]["primary_predictor"]),
            "outcome_join_performed": False,
            "event_ids_frozen": True,
        }
    )
    Path(args.output_dir, MANIFEST_OUTPUT).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(outputs[GATE_OUTPUT].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
