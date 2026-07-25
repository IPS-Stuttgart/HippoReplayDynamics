#!/usr/bin/env python3
"""Compare detected SWR/replay events with promoted off-SWR candidates."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from aggregate_all_session_model_evidence import (
    DEFAULT_FIRST_ORDER_IMM_MODEL,
    DEFAULT_MARGIN_POSITIVE_MODEL,
    DEFAULT_MARGIN_REFERENCE_MODEL,
    DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
    DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS,
    DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
)

STATIONARY_MODEL = "sorted-spike-state-space-stationary"
FRAGMENTED_MODEL = "sorted-spike-state-space-fragmented"

DETECTED_REPLAY_CLASS = "detected_replay_or_swr"
PROMOTED_OFF_SWR_CLASS = "promoted_off_swr"
REJECTED_HIGH_SPECIFICITY_CLASS = "rejected_high_specificity_off_swr_candidates"

LOGZ_COLUMNS = {
    STATIONARY_MODEL: "logZ_stationary",
    DEFAULT_MARGIN_REFERENCE_MODEL: "logZ_diffusion",
    FRAGMENTED_MODEL: "logZ_fragmented",
    DEFAULT_FIRST_ORDER_IMM_MODEL: "logZ_first_order_imm",
    DEFAULT_MARGIN_POSITIVE_MODEL: "logZ_momentum_exact_sparse",
}

COMPARISON_COLUMNS = (
    "event_class",
    "session",
    "rat",
    "event_index",
    "candidate_id",
    "window_role",
    "null_index",
    "candidate_rank",
    "window_start_s",
    "window_end_s",
    "duration_s",
    "n_spikes",
    "active_cell_count",
    "mean_speed_cm_s",
    "animal_speed_median",
    "animal_speed_max",
    "run_or_immobility_state",
    "nearest_known_swr_distance_s",
    "overlaps_known_swr",
    "trajectory_posterior_entropy",
    "decoded_path_length",
    "decoded_speed",
    "decoded_endpoint_distance",
    "decoded_start_to_end_distance",
    "required_models_present",
    "required_models_total",
    "required_models_complete",
    "missing_required_models",
    "margin_threshold",
    "best_exact_trajectory_model",
    "best_trajectory_log_evidence",
    "best_nontrajectory_model",
    "best_nontrajectory_log_evidence",
    "trajectory_minus_nontrajectory_margin",
    "trajectory_raw_win",
    "trajectory_confident_claim",
    "nontrajectory_confident_claim",
    "margin_decision",
    *LOGZ_COLUMNS.values(),
)

MODEL_WINNER_COLUMNS = (
    "event_class",
    "best_exact_trajectory_model",
    "events",
    "fraction_of_event_class",
    "model_rank",
    "is_first_order_imm",
    "is_exact_sparse_momentum",
)

FAMILY_MARGIN_COLUMNS = (
    "event_class",
    "events",
    "required_complete_events",
    "incomplete_core_events",
    "trajectory_raw_wins",
    "nontrajectory_raw_wins",
    "trajectory_raw_win_fraction",
    "trajectory_confident_claims",
    "nontrajectory_confident_claims",
    "ambiguous_events",
    "trajectory_confident_claim_fraction",
    "nontrajectory_confident_claim_fraction",
    "mean_trajectory_minus_nontrajectory_margin",
    "median_trajectory_minus_nontrajectory_margin",
    "min_trajectory_minus_nontrajectory_margin",
    "max_trajectory_minus_nontrajectory_margin",
    "first_order_imm_best_events",
    "first_order_imm_best_fraction",
    "exact_sparse_momentum_best_events",
    "exact_sparse_momentum_best_fraction",
    "most_common_best_exact_trajectory_model",
)

RAT_SESSION_COLUMNS = (
    "event_class",
    "rat",
    "session",
    "events",
    "trajectory_confident_claims",
    "nontrajectory_confident_claims",
    "trajectory_confident_claim_fraction",
    "median_trajectory_minus_nontrajectory_margin",
    "min_trajectory_minus_nontrajectory_margin",
    "first_order_imm_best_events",
    "exact_sparse_momentum_best_events",
    "immobile_events",
    "running_events",
    "median_mean_speed_cm_s",
)

BEHAVIOR_COLUMNS = (
    "event_class",
    "events",
    "immobile_events",
    "running_events",
    "unknown_speed_events",
    "immobile_fraction",
    "running_fraction",
    "mean_mean_speed_cm_s",
    "median_mean_speed_cm_s",
    "max_mean_speed_cm_s",
    "median_animal_speed_median",
    "median_animal_speed_max",
    "median_n_spikes",
    "median_active_cell_count",
    "median_duration_s",
    "median_nearest_known_swr_distance_s",
    "min_nearest_known_swr_distance_s",
    "overlaps_known_swr_events",
)

GATE_COLUMNS = ("gate", "passed", "observed", "criterion", "required_for_overall")


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required input table is missing: {path}")
    return pd.read_csv(path)


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _bool_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", "", "nan", "none"}
    return normalized.map(lambda value: True if value in true_values else False if value in false_values else default)


def _bool_value(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "", "nan", "none"}:
        return False
    return default


def _first_present_numeric(group: pd.DataFrame, columns: Iterable[str], *, default: float = np.nan) -> float:
    for column in columns:
        if column not in group:
            continue
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return float(default)


def _first_present_value(group: pd.DataFrame, columns: Iterable[str], *, default: object = "") -> object:
    for column in columns:
        if column not in group:
            continue
        values = group[column].dropna()
        if not values.empty:
            return values.iloc[0]
    return default


def _safe_fraction(value: float, denominator: float) -> float:
    if denominator is None or float(denominator) == 0.0:
        return np.nan
    return float(value) / float(denominator)


def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else np.nan


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def _safe_min(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.min()) if not numeric.empty else np.nan


def _safe_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else np.nan


def _rat_from_session(value: object) -> str:
    return str(value).split("/", 1)[0]


def _add_missing_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out:
            out[column] = np.nan
    return out


def _family_margin_decisions(
    frame: pd.DataFrame,
    *,
    group_cols: tuple[str, ...],
    required_models: tuple[str, ...] = DEFAULT_PAPER_REQUIRED_FULL_CORE_MODELS,
    trajectory_models: tuple[str, ...] = DEFAULT_PAPER_EXACT_TRAJECTORY_MODELS,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    required = tuple(str(model) for model in required_models)
    required_set = set(required)
    trajectory_set = set(str(model) for model in trajectory_models)
    columns = [
        *group_cols,
        "required_models_present",
        "required_models_total",
        "required_models_complete",
        "missing_required_models",
        "margin_threshold",
        "best_exact_trajectory_model",
        "best_trajectory_log_evidence",
        "best_nontrajectory_model",
        "best_nontrajectory_log_evidence",
        "trajectory_minus_nontrajectory_margin",
        "trajectory_raw_win",
        "trajectory_confident_claim",
        "nontrajectory_confident_claim",
        "margin_decision",
    ]
    if frame.empty or not set(("model", "log_evidence", *group_cols)).issubset(frame.columns):
        return pd.DataFrame(columns=columns)

    status_ok = frame["status"].astype(str).eq("success") if "status" in frame else pd.Series(True, index=frame.index)
    comparable = _bool_series(frame, "evidence_comparable", default=True)
    ok = frame[status_ok & comparable].copy()
    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(list(group_cols), sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        core = group[group["model"].astype(str).isin(required_set)].dropna(subset=["log_evidence"]).copy()
        present = tuple(model for model in required if model in set(core["model"].astype(str)))
        missing = tuple(model for model in required if model not in set(present))
        trajectory = core[core["model"].astype(str).isin(trajectory_set)]
        nontrajectory = core[~core["model"].astype(str).isin(trajectory_set)]
        if trajectory.empty or nontrajectory.empty:
            best_trajectory_model = ""
            best_trajectory_value = np.nan
            best_nontrajectory_model = ""
            best_nontrajectory_value = np.nan
            margin = np.nan
            raw_win = False
            trajectory_claim = False
            nontrajectory_claim = False
            decision = "incomplete_core"
        else:
            best_trajectory = trajectory.sort_values("log_evidence", ascending=False).iloc[0]
            best_nontrajectory = nontrajectory.sort_values("log_evidence", ascending=False).iloc[0]
            best_trajectory_model = str(best_trajectory["model"])
            best_trajectory_value = float(best_trajectory["log_evidence"])
            best_nontrajectory_model = str(best_nontrajectory["model"])
            best_nontrajectory_value = float(best_nontrajectory["log_evidence"])
            margin = best_trajectory_value - best_nontrajectory_value
            raw_win = bool(margin > 0.0)
            trajectory_claim = bool(not missing and margin >= float(margin_threshold))
            nontrajectory_claim = bool(not missing and margin <= -float(margin_threshold))
            if missing:
                decision = "incomplete_core"
            elif trajectory_claim:
                decision = "trajectory"
            elif nontrajectory_claim:
                decision = "nontrajectory"
            else:
                decision = "ambiguous"
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "required_models_present": len(present),
                "required_models_total": len(required),
                "required_models_complete": bool(not missing),
                "missing_required_models": " ".join(missing),
                "margin_threshold": float(margin_threshold),
                "best_exact_trajectory_model": best_trajectory_model,
                "best_trajectory_log_evidence": float(best_trajectory_value),
                "best_nontrajectory_model": best_nontrajectory_model,
                "best_nontrajectory_log_evidence": float(best_nontrajectory_value),
                "trajectory_minus_nontrajectory_margin": float(margin),
                "trajectory_raw_win": raw_win,
                "trajectory_confident_claim": trajectory_claim,
                "nontrajectory_confident_claim": nontrajectory_claim,
                "margin_decision": decision,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _metadata_table(frame: pd.DataFrame, *, group_cols: tuple[str, ...], event_class: str) -> pd.DataFrame:
    columns = [
        *group_cols,
        "rat",
        "candidate_rank",
        "window_start_s",
        "window_end_s",
        "duration_s",
        "n_spikes",
        "active_cell_count",
        "mean_speed_cm_s",
        "animal_speed_median",
        "animal_speed_max",
        "run_or_immobility_state",
        "nearest_known_swr_distance_s",
        "overlaps_known_swr",
        "trajectory_posterior_entropy",
        "decoded_path_length",
        "decoded_speed",
        "decoded_endpoint_distance",
        "decoded_start_to_end_distance",
    ]
    if frame.empty or not set(group_cols).issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(list(group_cols), sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        start = _first_present_numeric(group, ("window_start_s", "real_event_start_s", "event_start_s", "start_s"))
        end = _first_present_numeric(group, ("window_end_s", "real_event_end_s", "event_end_s", "end_s"))
        duration = _first_present_numeric(
            group,
            ("duration_s", "window_duration_s", "real_event_duration_s", "event_duration_s"),
        )
        if not np.isfinite(duration) and np.isfinite(start) and np.isfinite(end):
            duration = float(end - start)
        rat = _first_present_value(group, ("rat",), default=_rat_from_session(group["session"].iloc[0]))
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "rat": str(rat),
                "candidate_rank": _first_present_numeric(group, ("candidate_rank",)),
                "window_start_s": start,
                "window_end_s": end,
                "duration_s": duration,
                "n_spikes": _first_present_numeric(group, ("n_spikes", "null_n_spikes", "real_n_spikes")),
                "active_cell_count": _first_present_numeric(
                    group,
                    ("active_cell_count", "null_active_cell_count", "real_active_cell_count"),
                ),
                "mean_speed_cm_s": _first_present_numeric(group, ("mean_speed_cm_s", "animal_speed_mean")),
                "animal_speed_median": _first_present_numeric(group, ("animal_speed_median",)),
                "animal_speed_max": _first_present_numeric(group, ("animal_speed_max",)),
                "run_or_immobility_state": str(_first_present_value(group, ("run_or_immobility_state",), default="")),
                "nearest_known_swr_distance_s": _first_present_numeric(
                    group,
                    ("nearest_known_swr_distance_s", "distance_to_nearest_swr_s"),
                    default=0.0 if event_class == DETECTED_REPLAY_CLASS else np.nan,
                ),
                "overlaps_known_swr": _bool_value(
                    _first_present_value(
                        group,
                        ("overlaps_known_swr",),
                        default=True if event_class == DETECTED_REPLAY_CLASS else False,
                    ),
                    default=True if event_class == DETECTED_REPLAY_CLASS else False,
                ),
                "trajectory_posterior_entropy": _first_present_numeric(group, ("trajectory_posterior_entropy",)),
                "decoded_path_length": _first_present_numeric(group, ("decoded_path_length",)),
                "decoded_speed": _first_present_numeric(group, ("decoded_speed",)),
                "decoded_endpoint_distance": _first_present_numeric(group, ("decoded_endpoint_distance",)),
                "decoded_start_to_end_distance": _first_present_numeric(group, ("decoded_start_to_end_distance",)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _logz_table(frame: pd.DataFrame, *, group_cols: tuple[str, ...]) -> pd.DataFrame:
    columns = [*group_cols, *LOGZ_COLUMNS.values()]
    if frame.empty or not set(("model", "log_evidence", *group_cols)).issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    subset = frame[frame["model"].astype(str).isin(LOGZ_COLUMNS)].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    pivot = subset.pivot_table(index=list(group_cols), columns="model", values="log_evidence", aggfunc="max")
    pivot = pivot.rename(columns=LOGZ_COLUMNS).reset_index()
    for column in LOGZ_COLUMNS.values():
        if column not in pivot:
            pivot[column] = np.nan
    return pivot[columns]


def _normalize_decision_table(
    *,
    frame: pd.DataFrame,
    group_cols: tuple[str, ...],
    event_class: str,
    margin_threshold: float,
    precomputed_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    decisions = (
        precomputed_decisions.copy()
        if precomputed_decisions is not None and not precomputed_decisions.empty
        else _family_margin_decisions(frame, group_cols=group_cols, margin_threshold=margin_threshold)
    )
    if decisions.empty:
        return pd.DataFrame(columns=list(COMPARISON_COLUMNS))
    if "best_trajectory_model" in decisions and "best_exact_trajectory_model" not in decisions:
        decisions = decisions.rename(columns={"best_trajectory_model": "best_exact_trajectory_model"})
    if "trajectory_minus_nontrajectory_log_evidence" in decisions and "trajectory_minus_nontrajectory_margin" not in decisions:
        decisions = decisions.rename(columns={"trajectory_minus_nontrajectory_log_evidence": "trajectory_minus_nontrajectory_margin"})
    if "trajectory_raw_win" not in decisions and "trajectory_minus_nontrajectory_margin" in decisions:
        decisions["trajectory_raw_win"] = pd.to_numeric(
            decisions["trajectory_minus_nontrajectory_margin"],
            errors="coerce",
        ).gt(0.0)
    decisions = _add_missing_columns(decisions, group_cols)
    metadata = _metadata_table(frame, group_cols=group_cols, event_class=event_class)
    logz = _logz_table(frame, group_cols=group_cols)
    out = decisions.merge(metadata, on=list(group_cols), how="left", suffixes=("", "_from_scores"))
    for column in metadata.columns:
        fallback = f"{column}_from_scores"
        if fallback in out:
            if column in out:
                out[column] = out[column].combine_first(out[fallback])
            else:
                out[column] = out[fallback]
            out = out.drop(columns=[fallback])
    out = out.merge(logz, on=list(group_cols), how="left")
    out["event_class"] = event_class
    if "rat" not in out:
        out["rat"] = out["session"].map(_rat_from_session)
    else:
        out["rat"] = out["rat"].fillna(out["session"].map(_rat_from_session)).astype(str)
    if "window_role" not in out:
        out["window_role"] = event_class
    if "null_index" not in out:
        out["null_index"] = np.nan
    out["candidate_id"] = out.apply(_candidate_id, axis=1)
    for column in COMPARISON_COLUMNS:
        if column not in out:
            out[column] = np.nan
    return out[list(COMPARISON_COLUMNS)]


def _candidate_id(row: pd.Series) -> str:
    session = str(row.get("session", ""))
    event_index = row.get("event_index", "")
    null_index = row.get("null_index", np.nan)
    if pd.notna(null_index):
        try:
            null_text = str(int(float(null_index)))
        except (TypeError, ValueError):
            null_text = str(null_index)
        return f"{session}|event={event_index}|null={null_text}"
    return f"{session}|event={event_index}"


def build_comparison_table(
    *,
    swr_event_model_evidence: pd.DataFrame,
    off_swr_event_model_evidence: pd.DataFrame,
    off_swr_decisions: pd.DataFrame | None = None,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    swr = _normalize_decision_table(
        frame=swr_event_model_evidence,
        group_cols=("session", "event_index"),
        event_class=DETECTED_REPLAY_CLASS,
        margin_threshold=margin_threshold,
    )
    off = _normalize_decision_table(
        frame=off_swr_event_model_evidence,
        group_cols=("session", "event_index", "window_role", "null_index"),
        event_class=PROMOTED_OFF_SWR_CLASS,
        margin_threshold=margin_threshold,
        precomputed_decisions=off_swr_decisions,
    )
    comparison = pd.concat([swr, off], ignore_index=True)
    if comparison.empty:
        return pd.DataFrame(columns=list(COMPARISON_COLUMNS))
    comparison["event_index"] = pd.to_numeric(comparison["event_index"], errors="coerce").astype("Int64")
    comparison["null_index"] = pd.to_numeric(comparison["null_index"], errors="coerce").astype("Int64")
    comparison = comparison.sort_values(["event_class", "session", "event_index", "null_index"], kind="mergesort")
    return comparison.reset_index(drop=True)


def model_winner_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(columns=list(MODEL_WINNER_COLUMNS))
    rows: list[dict[str, object]] = []
    for event_class, group in comparison.groupby("event_class", sort=True):
        best = group["best_exact_trajectory_model"].fillna("").astype(str).str.strip()
        counts = best[best.ne("")].value_counts()
        total = len(group)
        for rank, (model, count) in enumerate(counts.items(), start=1):
            rows.append(
                {
                    "event_class": event_class,
                    "best_exact_trajectory_model": model,
                    "events": int(count),
                    "fraction_of_event_class": _safe_fraction(int(count), total),
                    "model_rank": int(rank),
                    "is_first_order_imm": bool(model == DEFAULT_FIRST_ORDER_IMM_MODEL),
                    "is_exact_sparse_momentum": bool(model == DEFAULT_MARGIN_POSITIVE_MODEL),
                }
            )
    return pd.DataFrame(rows, columns=list(MODEL_WINNER_COLUMNS))


def family_margin_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(columns=list(FAMILY_MARGIN_COLUMNS))
    rows: list[dict[str, object]] = []
    for event_class, group in comparison.groupby("event_class", sort=True):
        margins = pd.to_numeric(group["trajectory_minus_nontrajectory_margin"], errors="coerce")
        complete = _bool_series(group, "required_models_complete")
        trajectory_claim = _bool_series(group, "trajectory_confident_claim")
        nontrajectory_claim = _bool_series(group, "nontrajectory_confident_claim")
        best = group["best_exact_trajectory_model"].fillna("").astype(str).str.strip()
        reported_best = best[best.ne("")]
        events = len(group)
        rows.append(
            {
                "event_class": event_class,
                "events": events,
                "required_complete_events": int(complete.sum()),
                "incomplete_core_events": int((group["margin_decision"].astype(str) == "incomplete_core").sum()),
                "trajectory_raw_wins": int((margins > 0.0).sum()),
                "nontrajectory_raw_wins": int((margins < 0.0).sum()),
                "trajectory_raw_win_fraction": float((margins > 0.0).mean()) if events else np.nan,
                "trajectory_confident_claims": int(trajectory_claim.sum()),
                "nontrajectory_confident_claims": int(nontrajectory_claim.sum()),
                "ambiguous_events": int((group["margin_decision"].astype(str) == "ambiguous").sum()),
                "trajectory_confident_claim_fraction": _safe_fraction(int(trajectory_claim.sum()), events),
                "nontrajectory_confident_claim_fraction": _safe_fraction(int(nontrajectory_claim.sum()), events),
                "mean_trajectory_minus_nontrajectory_margin": _safe_mean(margins),
                "median_trajectory_minus_nontrajectory_margin": _safe_median(margins),
                "min_trajectory_minus_nontrajectory_margin": _safe_min(margins),
                "max_trajectory_minus_nontrajectory_margin": _safe_max(margins),
                "first_order_imm_best_events": int(best.eq(DEFAULT_FIRST_ORDER_IMM_MODEL).sum()),
                "first_order_imm_best_fraction": _safe_fraction(int(best.eq(DEFAULT_FIRST_ORDER_IMM_MODEL).sum()), events),
                "exact_sparse_momentum_best_events": int(best.eq(DEFAULT_MARGIN_POSITIVE_MODEL).sum()),
                "exact_sparse_momentum_best_fraction": _safe_fraction(int(best.eq(DEFAULT_MARGIN_POSITIVE_MODEL).sum()), events),
                "most_common_best_exact_trajectory_model": ("" if reported_best.empty else str(reported_best.value_counts().index[0])),
            }
        )
    return pd.DataFrame(rows, columns=list(FAMILY_MARGIN_COLUMNS))


def rat_session_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(columns=list(RAT_SESSION_COLUMNS))
    rows: list[dict[str, object]] = []
    for key, group in comparison.groupby(["event_class", "rat", "session"], sort=True):
        event_class, rat, session = key
        events = len(group)
        best = group["best_exact_trajectory_model"].fillna("").astype(str)
        rows.append(
            {
                "event_class": event_class,
                "rat": rat,
                "session": session,
                "events": events,
                "trajectory_confident_claims": int(_bool_series(group, "trajectory_confident_claim").sum()),
                "nontrajectory_confident_claims": int(_bool_series(group, "nontrajectory_confident_claim").sum()),
                "trajectory_confident_claim_fraction": _safe_fraction(
                    int(_bool_series(group, "trajectory_confident_claim").sum()),
                    events,
                ),
                "median_trajectory_minus_nontrajectory_margin": _safe_median(group["trajectory_minus_nontrajectory_margin"]),
                "min_trajectory_minus_nontrajectory_margin": _safe_min(group["trajectory_minus_nontrajectory_margin"]),
                "first_order_imm_best_events": int(best.eq(DEFAULT_FIRST_ORDER_IMM_MODEL).sum()),
                "exact_sparse_momentum_best_events": int(best.eq(DEFAULT_MARGIN_POSITIVE_MODEL).sum()),
                "immobile_events": int(group["run_or_immobility_state"].astype(str).eq("immobile").sum()),
                "running_events": int(group["run_or_immobility_state"].astype(str).eq("run").sum()),
                "median_mean_speed_cm_s": _safe_median(group["mean_speed_cm_s"]),
            }
        )
    return pd.DataFrame(rows, columns=list(RAT_SESSION_COLUMNS))


def _behavior_group_row(event_class: str, group: pd.DataFrame) -> dict[str, object]:
    events = len(group)
    state = group.get("run_or_immobility_state", pd.Series("", index=group.index)).astype(str)
    speed = _numeric(group, "mean_speed_cm_s")
    nearest = _numeric(group, "nearest_known_swr_distance_s")
    overlaps = _bool_series(group, "overlaps_known_swr")
    immobile = int(state.eq("immobile").sum())
    running = int(state.eq("run").sum())
    unknown = int((state.eq("") | state.eq("nan") | state.eq("unknown_speed")).sum())
    return {
        "event_class": event_class,
        "events": events,
        "immobile_events": immobile,
        "running_events": running,
        "unknown_speed_events": unknown,
        "immobile_fraction": _safe_fraction(immobile, events),
        "running_fraction": _safe_fraction(running, events),
        "mean_mean_speed_cm_s": _safe_mean(speed),
        "median_mean_speed_cm_s": _safe_median(speed),
        "max_mean_speed_cm_s": _safe_max(speed),
        "median_animal_speed_median": _safe_median(group.get("animal_speed_median", pd.Series(dtype=float))),
        "median_animal_speed_max": _safe_median(group.get("animal_speed_max", pd.Series(dtype=float))),
        "median_n_spikes": _safe_median(group.get("n_spikes", pd.Series(dtype=float))),
        "median_active_cell_count": _safe_median(group.get("active_cell_count", pd.Series(dtype=float))),
        "median_duration_s": _safe_median(group.get("duration_s", pd.Series(dtype=float))),
        "median_nearest_known_swr_distance_s": _safe_median(nearest),
        "min_nearest_known_swr_distance_s": _safe_min(nearest),
        "overlaps_known_swr_events": int(overlaps.sum()),
    }


def _rejected_high_specificity_rows(high_specificity: pd.DataFrame) -> pd.DataFrame:
    if high_specificity.empty:
        return pd.DataFrame()
    table = high_specificity.copy()
    if "passes_high_specificity_promotion_filter" in table:
        table = table[~_bool_series(table, "passes_high_specificity_promotion_filter")].copy()
    elif "high_specificity_label" in table:
        table = table[~table["high_specificity_label"].astype(str).eq("promotion_ready_high_specificity_candidate")].copy()
    if table.empty:
        return table
    rename = {
        "distance_to_nearest_swr_s": "nearest_known_swr_distance_s",
        "animal_speed_mean": "mean_speed_cm_s",
    }
    return table.rename(columns={key: value for key, value in rename.items() if key in table})


def behavior_summary(comparison: pd.DataFrame, high_specificity_candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not comparison.empty:
        for event_class, group in comparison.groupby("event_class", sort=True):
            rows.append(_behavior_group_row(str(event_class), group))
    rejected = _rejected_high_specificity_rows(high_specificity_candidates if high_specificity_candidates is not None else pd.DataFrame())
    if not rejected.empty:
        rows.append(_behavior_group_row(REJECTED_HIGH_SPECIFICITY_CLASS, rejected))
    return pd.DataFrame(rows, columns=list(BEHAVIOR_COLUMNS))


def gate_summary(comparison: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str, *, required: bool = True) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "required_for_overall": bool(required),
            }
        )

    summary = family_margin_summary(comparison).set_index("event_class") if not comparison.empty else pd.DataFrame()
    swr = summary.loc[DETECTED_REPLAY_CLASS] if DETECTED_REPLAY_CLASS in summary.index else None
    off = summary.loc[PROMOTED_OFF_SWR_CLASS] if PROMOTED_OFF_SWR_CLASS in summary.index else None
    swr_events = int(swr["events"]) if swr is not None else 0
    off_events = int(off["events"]) if off is not None else 0
    add("swr_events_present", swr_events > 0, swr_events, "detected replay/SWR events > 0")
    add("promoted_off_swr_candidates_present", off_events > 0, off_events, "promoted off-SWR candidates > 0")
    swr_claim_fraction = float(swr["trajectory_confident_claim_fraction"]) if swr is not None else np.nan
    add(
        "swr_trajectory_confident_majority",
        bool(np.isfinite(swr_claim_fraction) and swr_claim_fraction > 0.5),
        swr_claim_fraction,
        "trajectory confident claim fraction > 0.5",
    )
    add(
        "off_swr_trajectory_confident_all",
        bool(off_events > 0 and int(off["trajectory_confident_claims"]) == off_events) if off is not None else False,
        f"{0 if off is None else int(off['trajectory_confident_claims'])}/{off_events}",
        "all promoted off-SWR candidates are trajectory-confident",
    )
    swr_fo_fraction = float(swr["first_order_imm_best_fraction"]) if swr is not None else np.nan
    off_fo_fraction = float(off["first_order_imm_best_fraction"]) if off is not None else np.nan
    add(
        "first_order_imm_dominates_both_classes",
        bool(np.isfinite(swr_fo_fraction) and np.isfinite(off_fo_fraction) and swr_fo_fraction > 0.5 and off_fo_fraction > 0.5),
        f"swr={swr_fo_fraction}; off_swr={off_fo_fraction}",
        "first-order IMM is best exact trajectory model for > 50% in both classes",
    )
    behavior_index = behavior.set_index("event_class") if not behavior.empty else pd.DataFrame()
    if PROMOTED_OFF_SWR_CLASS in behavior_index.index:
        off_behavior = behavior_index.loc[PROMOTED_OFF_SWR_CLASS]
        off_immobile = int(off_behavior["immobile_events"])
        off_running = int(off_behavior["running_events"])
        off_min_distance = float(off_behavior["min_nearest_known_swr_distance_s"])
    else:
        off_immobile = 0
        off_running = 0
        off_min_distance = np.nan
    add(
        "off_swr_candidates_all_immobile",
        bool(off_events > 0 and off_immobile == off_events and off_running == 0),
        f"immobile={off_immobile}/{off_events}; running={off_running}",
        "all promoted off-SWR candidates are immobile",
    )
    add(
        "off_swr_candidates_distant_from_known_swrs",
        bool(np.isfinite(off_min_distance) and off_min_distance >= 1.0),
        off_min_distance,
        "minimum nearest known SWR distance >= 1 s",
    )
    if REJECTED_HIGH_SPECIFICITY_CLASS in behavior_index.index:
        rejected = behavior_index.loc[REJECTED_HIGH_SPECIFICITY_CLASS]
        rejected_events = int(rejected["events"])
        rejected_running = int(rejected["running_events"])
        add(
            "rejected_high_specificity_behavior_reported",
            rejected_events > 0,
            f"running={rejected_running}/{rejected_events}",
            "rejected high-specificity candidates included in behavior summary",
            required=False,
        )
    required_rows = [row for row in rows if row["required_for_overall"]]
    add(
        "overall",
        all(row["passed"] for row in required_rows),
        f"{sum(row['passed'] for row in required_rows)}/{len(required_rows)} required gates passed",
        "all required gates pass",
        required=True,
    )
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def write_swr_off_swr_dynamics_outputs(
    *,
    swr_event_model_evidence: Path,
    off_swr_event_model_evidence: Path,
    output: Path,
    off_swr_decisions: Path | None = None,
    off_swr_high_specificity_candidates: Path | None = None,
    margin_threshold: float = DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    swr_scores = _read_required_csv(swr_event_model_evidence)
    off_scores = _read_required_csv(off_swr_event_model_evidence)
    off_decisions = _read_optional_csv(off_swr_decisions)
    high_specificity = _read_optional_csv(off_swr_high_specificity_candidates)
    output.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison_table(
        swr_event_model_evidence=swr_scores,
        off_swr_event_model_evidence=off_scores,
        off_swr_decisions=off_decisions,
        margin_threshold=margin_threshold,
    )
    outputs = {
        "swr_off_swr_dynamics_comparison.csv": comparison,
        "swr_off_swr_model_winner_summary.csv": model_winner_summary(comparison),
        "swr_off_swr_family_margin_summary.csv": family_margin_summary(comparison),
        "swr_off_swr_rat_session_summary.csv": rat_session_summary(comparison),
    }
    outputs["swr_off_swr_behavior_summary.csv"] = behavior_summary(comparison, high_specificity)
    outputs["swr_off_swr_gate_summary.csv"] = gate_summary(comparison, outputs["swr_off_swr_behavior_summary.csv"])
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swr-event-model-evidence", required=True, type=Path)
    parser.add_argument("--off-swr-event-model-evidence", required=True, type=Path)
    parser.add_argument("--off-swr-decisions", type=Path)
    parser.add_argument("--off-swr-high-specificity-candidates", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MOMENTUM_CONFIDENCE_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = write_swr_off_swr_dynamics_outputs(
        swr_event_model_evidence=args.swr_event_model_evidence,
        off_swr_event_model_evidence=args.off_swr_event_model_evidence,
        off_swr_decisions=args.off_swr_decisions,
        off_swr_high_specificity_candidates=args.off_swr_high_specificity_candidates,
        output=args.output,
        margin_threshold=args.margin_threshold,
    )
    gate = outputs["swr_off_swr_gate_summary.csv"]
    print(gate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
