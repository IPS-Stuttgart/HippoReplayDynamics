#!/usr/bin/env python3
"""Discover replay-like trajectory-family candidates in spike-matched off-SWR windows.

This is a post-hoc discovery layer on top of the existing spike-matched
off-SWR null scorer. It treats those scored windows as candidate observations,
applies the same exact trajectory-family versus static/nontrajectory gate used
for controls, removes windows that are not marked as off-SWR, clusters detected
candidates by session/time, and summarizes behavior/LFP covariates when those
columns are available.
"""

from __future__ import annotations

import argparse
import glob
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_event_window_sensitivity import DEFAULT_MARGIN_THRESHOLD
from spike_matched_event_window_null import (
    FULL_CORE_REQUIRED_MODELS,
    LIGHTWEIGHT_FO_IMM_STATIONARY_SCOPE,
    matched_null_family_margin_decisions,
    resolve_family_model_sets,
)


DEFAULT_OFF_SWR_DISCOVERY_MODELS = (
    "sorted-spike-state-space-stationary",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
)

DEFAULT_NEAREST_SWR_EXCLUSION_RADII_S = (0.1, 0.25, 0.5, 1.0)
DEFAULT_CANDIDATE_TIER_THRESHOLDS = (
    ("weak", DEFAULT_MARGIN_THRESHOLD),
    ("moderate", 20.0),
    ("strong", 50.0),
    ("extreme", 100.0),
)

DEFAULT_BEHAVIOR_LFP_COLUMNS = (
    "window_duration_s",
    "n_spikes",
    "active_cell_count",
    "real_n_spikes",
    "n_spikes_delta",
    "n_spikes_relative_delta",
    "pre_event_rate",
    "ripple_power",
    "theta_metric",
    "speed_pre",
    "speed_post",
    "event_duration",
    "spike_count",
    "distance_to_goal",
    "distance_to_well",
    "novelty",
    "time_in_session",
    "sleep_state",
    "lfp_power",
    "ripple_band_power",
)

TRIAGE_MEAN_SPEED_COLUMNS = ("animal_speed_mean", "speed_mean", "speed_pre")
TRIAGE_MEDIAN_SPEED_COLUMNS = ("animal_speed_median", "speed_median", "speed_pre")
TRIAGE_MAX_SPEED_COLUMNS = ("animal_speed_max", "speed_max", "speed_post", "speed_pre")
TRIAGE_ENTROPY_COLUMNS = (
    "diagnostic_mean_trajectory_posterior_entropy",
    "mean_trajectory_posterior_entropy",
    "diagnostic_terminal_posterior_entropy",
    "terminal_posterior_entropy",
)
TRIAGE_PATH_LENGTH_COLUMNS = (
    "diagnostic_replay_posterior_mean_path_length_cm",
    "diagnostic_posterior_mean_path_length_cm",
    "posterior_mean_path_length_cm",
    "posterior_mean_path_length",
    "diagnostic_map_path_length_cm",
    "map_path_length_cm",
    "map_path_length",
)
TRIAGE_ENDPOINT_X_COLUMNS = ("diagnostic_decoded_endpoint_x", "decoded_endpoint_x")
TRIAGE_ENDPOINT_Y_COLUMNS = ("diagnostic_decoded_endpoint_y", "decoded_endpoint_y")
TRIAGE_START_X_COLUMNS = (
    "diagnostic_decoded_start_x",
    "decoded_start_x",
    "diagnostic_decoded_map_x",
    "decoded_map_x",
)
TRIAGE_START_Y_COLUMNS = (
    "diagnostic_decoded_start_y",
    "decoded_start_y",
    "diagnostic_decoded_map_y",
    "decoded_map_y",
)
TRIAGE_ANIMAL_X_COLUMNS = ("animal_x", "position_x", "current_x", "window_mean_x", "mean_x")
TRIAGE_ANIMAL_Y_COLUMNS = ("animal_y", "position_y", "current_y", "window_mean_y", "mean_y")

KEY_COLUMNS = ("session", "event_index", "window_role", "null_index")
TRAJECTORY_CANDIDATE_CLASS = "off_swr_trajectory_family_candidate"
STATIC_NONTRAJECTORY_CLASS = "off_swr_static_nontrajectory"
AMBIGUOUS_CLASS = "ambiguous"
INCOMPLETE_CLASS = "incomplete_core"
EXCLUDED_SWR_OVERLAP_CLASS = "excluded_known_swr_overlap"
INTERESTING_CANDIDATE_LABEL = "interesting_off_swr_trajectory_candidate"
MOVEMENT_SPIKING_LIKE_LABEL = "ordinary_movement_or_spiking_like"
LOW_INFORMATION_LABEL = "low_information_candidate"

DECISION_OUTPUT_COLUMNS = (
    "rat",
    "session",
    "event_index",
    "window_role",
    "null_index",
    "candidate_class",
    "is_trajectory_family_candidate",
    "passes_known_swr_exclusion",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "n_spikes",
    "n_time",
    "active_cell_count",
    "comparison_scope",
    "required_models_present",
    "required_models_total",
    "required_models_complete",
    "missing_required_models",
    "margin_threshold",
    "best_trajectory_model",
    "best_trajectory_log_evidence",
    "best_nontrajectory_model",
    "best_nontrajectory_log_evidence",
    "trajectory_minus_nontrajectory_log_evidence",
    "trajectory_minus_nontrajectory_log_evidence_per_spike",
    "trajectory_minus_nontrajectory_log_evidence_per_time_bin",
    "trajectory_confident_claim",
    "nontrajectory_confident_claim",
    "margin_decision",
)

CANDIDATE_TABLE_COLUMNS = (
    "candidate_rank",
    "candidate_specificity_label",
    "candidate_priority_score",
    "ordinary_movement_spiking_score",
    "session",
    "rat",
    "event_index",
    "null_index",
    "window_start_s",
    "window_end_s",
    "duration_s",
    "n_spikes",
    "active_cell_count",
    "trajectory_family_margin",
    "candidate_tier",
    "best_trajectory_model",
    "trajectory_confidence",
    "trajectory_posterior_entropy",
    "distance_to_nearest_swr_s",
    "overlaps_known_swr",
    "animal_speed_mean",
    "animal_speed_median",
    "animal_speed_max",
    "position_sample_count",
    "run_or_immobility_state",
    "decoded_path_length",
    "decoded_speed",
    "decoded_endpoint_distance",
    "decoded_start_to_end_distance",
    "candidate_cluster_id",
    "comparison_scope",
    "margin_threshold",
    "best_nontrajectory_model",
    "trajectory_margin_per_spike",
    "trajectory_margin_per_time_bin",
    "matched_null_rank",
    "template_event_index",
)

CANDIDATE_CLUSTER_TABLE_COLUMNS = (
    "rat",
    "session",
    "candidate_cluster_id",
    "cluster_index",
    "window_count",
    "template_event_count",
    "template_event_indices",
    "time_start_s",
    "time_end_s",
    "duration_s",
    "median_family_margin",
    "max_family_margin",
    "median_trajectory_confidence",
    "median_priority_score",
    "movement_spiking_like_windows",
    "interesting_candidate_windows",
    "best_trajectory_model",
    "best_candidate_event_index",
    "best_candidate_null_index",
    "median_n_spikes",
    "median_active_cell_count",
    "median_animal_speed_mean",
    "min_distance_to_nearest_swr_s",
)

CANDIDATE_GROUP_SUMMARY_COLUMNS = (
    "comparison_scope",
    "rat",
    "session",
    "candidate_windows",
    "candidate_clusters",
    "interesting_candidate_windows",
    "movement_spiking_like_windows",
    "low_information_candidate_windows",
    "median_candidate_priority_score",
    "median_family_margin",
    "median_trajectory_confidence",
    "median_distance_to_nearest_swr_s",
    "median_n_spikes",
    "median_active_cell_count",
    "median_animal_speed_mean",
)

CANDIDATE_VS_SWR_COLUMNS = (
    "comparison_scope",
    "off_swr_candidate_windows",
    "swr_reference_windows",
    "candidate_median_family_margin",
    "swr_median_family_margin",
    "candidate_minus_swr_median_family_margin",
    "candidate_median_trajectory_confidence",
    "swr_median_trajectory_confidence",
    "candidate_minus_swr_median_trajectory_confidence",
    "candidate_median_n_spikes",
    "swr_median_n_spikes",
    "candidate_minus_swr_median_n_spikes",
    "candidate_median_active_cell_count",
    "swr_median_active_cell_count",
    "candidate_minus_swr_median_active_cell_count",
    "candidate_median_trajectory_posterior_entropy",
    "swr_median_trajectory_posterior_entropy",
    "candidate_minus_swr_median_trajectory_posterior_entropy",
    "candidate_median_decoded_path_length",
    "swr_median_decoded_path_length",
    "candidate_minus_swr_median_decoded_path_length",
    "candidate_median_decoded_speed",
    "swr_median_decoded_speed",
    "candidate_minus_swr_median_decoded_speed",
    "candidate_median_duration_s",
    "swr_median_duration_s",
    "candidate_minus_swr_median_duration_s",
    "candidate_median_distance_to_nearest_swr_s",
    "swr_median_distance_to_nearest_swr_s",
    "candidate_median_animal_speed_mean",
    "swr_median_animal_speed_mean",
    "candidate_minus_swr_median_animal_speed_mean",
    "candidate_fraction_movement_spiking_like",
    "candidate_fraction_interesting",
    "candidate_fraction_run",
    "off_swr_best_trajectory_model_distribution",
    "swr_best_trajectory_model_distribution",
    "off_swr_vs_swr_interpretation",
    "claim_should_narrow",
)

CANDIDATE_VS_SWR_WINDOW_COLUMNS = (
    "comparison_scope",
    "window_set",
    "session",
    "rat",
    "event_index",
    "window_role",
    "null_index",
    "candidate_rank",
    "candidate_specificity_label",
    "candidate_cluster_id",
    "window_start_s",
    "window_end_s",
    "duration_s",
    "n_spikes",
    "active_cell_count",
    "trajectory_family_margin",
    "best_trajectory_model",
    "trajectory_confidence",
    "trajectory_posterior_entropy",
    "decoded_path_length",
    "decoded_speed",
    "decoded_endpoint_distance",
    "decoded_start_to_end_distance",
    "animal_speed_mean",
    "animal_speed_median",
    "animal_speed_max",
    "position_sample_count",
    "run_or_immobility_state",
    "distance_to_nearest_swr_s",
    "overlaps_known_swr",
)

CANDIDATE_VS_SWR_MODEL_DISTRIBUTION_COLUMNS = (
    "comparison_scope",
    "window_set",
    "best_trajectory_model",
    "windows",
    "fraction",
)

RUN_STATE_STRATIFIED_SUMMARY_COLUMNS = (
    "comparison_scope",
    "stratum",
    "window_set",
    "run_state",
    "windows",
    "trajectory_family_candidates",
    "candidate_fraction",
    "trajectory_confident_claims",
    "nontrajectory_confident_claims",
    "ambiguous_windows",
    "incomplete_windows",
    "mean_family_margin",
    "median_family_margin",
    "max_family_margin",
    "median_trajectory_confidence",
    "median_n_spikes",
    "median_active_cell_count",
    "median_trajectory_posterior_entropy",
    "median_decoded_path_length",
    "median_decoded_speed",
    "median_duration_s",
    "median_distance_to_nearest_swr_s",
    "median_animal_speed_mean",
    "best_trajectory_model_distribution",
)

RUN_STATE_SPECIFICITY_COLUMNS = (
    "comparison_scope",
    "off_swr_windows",
    "off_swr_immobile_windows",
    "off_swr_running_windows",
    "off_swr_unknown_speed_windows",
    "off_swr_candidates",
    "immobile_off_swr_candidates",
    "running_off_swr_candidates",
    "unknown_speed_off_swr_candidates",
    "immobile_candidate_fraction",
    "running_candidate_fraction",
    "unknown_speed_candidate_fraction",
    "swr_reference_windows",
    "immobile_candidate_signal_present",
    "run_state_specificity_interpretation",
    "claim_should_narrow_for_run_state",
)

NEAREST_SWR_EXCLUSION_COLUMNS = (
    "comparison_scope",
    "exclusion_radius_s",
    "exclusion_label",
    "off_swr_windows_before_exclusion",
    "candidate_windows_before_exclusion",
    "candidate_fraction_before_exclusion",
    "evaluable_distance_windows",
    "windows_after_exclusion",
    "candidate_windows_after_exclusion",
    "candidate_fraction_after_exclusion",
    "windows_excluded",
    "candidate_windows_excluded",
    "fraction_windows_retained",
    "fraction_candidates_retained",
    "candidate_sessions_after_exclusion",
    "candidate_rats_after_exclusion",
    "median_distance_to_nearest_swr_s_after_exclusion",
    "median_family_margin_after_exclusion",
    "mean_family_margin_after_exclusion",
    "median_candidate_family_margin_after_exclusion",
    "nearest_swr_exclusion_interpretation",
    "claim_should_narrow_for_nearest_swr",
)

NEAREST_SWR_SPECIFICITY_COLUMNS = (
    "comparison_scope",
    "off_swr_windows",
    "candidate_windows",
    "candidate_fraction",
    "evaluable_distance_windows",
    "candidate_windows_after_500ms_exclusion",
    "candidate_fraction_after_500ms_exclusion",
    "candidate_retention_after_500ms_exclusion",
    "candidate_windows_after_1s_exclusion",
    "candidate_fraction_after_1s_exclusion",
    "candidate_retention_after_1s_exclusion",
    "nearest_swr_specificity_interpretation",
    "claim_should_narrow_for_nearest_swr",
)

CANDIDATE_TIER_THRESHOLD_SUMMARY_COLUMNS = (
    "comparison_scope",
    "candidate_tier",
    "tier_margin_threshold",
    "off_swr_windows",
    "candidate_windows",
    "candidate_fraction",
    "candidate_sessions",
    "candidate_rats",
    "immobile_windows",
    "immobile_candidate_windows",
    "immobile_candidate_fraction",
    "running_windows",
    "running_candidate_windows",
    "running_candidate_fraction",
    "unknown_speed_windows",
    "unknown_speed_candidate_windows",
    "unknown_speed_candidate_fraction",
    "candidate_windows_after_500ms_swr_exclusion",
    "candidate_fraction_after_500ms_swr_exclusion",
    "candidate_windows_after_1s_swr_exclusion",
    "candidate_fraction_after_1s_swr_exclusion",
    "median_candidate_family_margin",
    "best_trajectory_model_distribution",
)

CANDIDATE_TIER_GROUP_SUMMARY_COLUMNS = (
    "comparison_scope",
    "rat",
    "session",
    "candidate_tier",
    "tier_margin_threshold",
    "off_swr_windows",
    "candidate_windows",
    "candidate_fraction",
    "immobile_candidate_windows",
    "running_candidate_windows",
    "unknown_speed_candidate_windows",
    "candidate_windows_after_500ms_swr_exclusion",
    "candidate_windows_after_1s_swr_exclusion",
    "median_candidate_family_margin",
)

CANDIDATE_TIER_NEAREST_SWR_EXCLUSION_COLUMNS = (
    "comparison_scope",
    "candidate_tier",
    "tier_margin_threshold",
    "exclusion_radius_s",
    "exclusion_label",
    "windows_after_exclusion",
    "candidate_windows_after_exclusion",
    "candidate_fraction_after_exclusion",
    "candidate_retention_after_exclusion",
    "median_candidate_family_margin_after_exclusion",
)

HIGH_SPECIFICITY_CANDIDATE_COLUMNS = (
    "candidate_rank",
    "candidate_specificity_label",
    "candidate_tier",
    "high_specificity_label",
    "session",
    "rat",
    "event_index",
    "null_index",
    "window_start_s",
    "window_end_s",
    "duration_s",
    "n_spikes",
    "active_cell_count",
    "trajectory_family_margin",
    "best_trajectory_model",
    "trajectory_confidence",
    "trajectory_posterior_entropy",
    "distance_to_nearest_swr_s",
    "run_or_immobility_state",
    "animal_speed_mean",
    "position_sample_count",
    "decoded_path_length",
    "decoded_speed",
    "passes_strong_tier",
    "passes_extreme_tier",
    "passes_500ms_swr_exclusion",
    "passes_1s_swr_exclusion",
    "speed_available",
    "passes_immobility_filter",
    "passes_specificity_label_filter",
    "passes_high_specificity_promotion_filter",
    "promotion_limitation",
)

PROMOTION_READINESS_COLUMNS = (
    "comparison_scope",
    "promotion_status",
    "promotion_ready",
    "off_swr_candidate_windows",
    "strong_candidate_windows",
    "extreme_candidate_windows",
    "strong_candidates_after_500ms_swr_exclusion",
    "strong_candidates_after_1s_swr_exclusion",
    "speed_evaluable_candidate_windows",
    "strong_immobile_candidate_windows",
    "high_specificity_candidate_windows",
    "nearest_swr_specificity_interpretation",
    "run_state_specificity_interpretation",
    "paper_claim_guidance",
)

SPEED_COVERAGE_COLUMNS = (
    "comparison_scope",
    "off_swr_windows",
    "off_swr_windows_with_position_samples",
    "off_swr_windows_with_speed",
    "off_swr_speed_coverage_fraction",
    "swr_reference_windows",
    "swr_reference_windows_with_speed",
    "candidate_windows",
    "candidate_windows_with_speed",
    "candidate_speed_coverage_fraction",
    "strong_candidate_windows",
    "strong_candidate_windows_with_speed",
    "strong_candidate_speed_coverage_fraction",
    "promotion_status",
    "speed_coverage_status",
    "speed_coverage_ready",
)

SUMMARY_COLUMNS = (
    "comparison_scope",
    "windows",
    "off_swr_windows",
    "required_complete_windows",
    "trajectory_family_candidates",
    "static_nontrajectory_windows",
    "ambiguous_windows",
    "incomplete_windows",
    "excluded_known_swr_overlap_windows",
    "candidate_fraction_of_off_swr",
    "candidate_sessions",
    "candidate_rats",
    "mean_family_margin",
    "median_family_margin",
    "mean_candidate_family_margin",
    "median_candidate_family_margin",
    "max_candidate_family_margin",
    "margin_threshold",
)

GROUP_SUMMARY_COLUMNS = (
    "comparison_scope",
    "rat",
    "session",
    "windows",
    "off_swr_windows",
    "trajectory_family_candidates",
    "static_nontrajectory_windows",
    "ambiguous_windows",
    "candidate_fraction_of_off_swr",
    "median_family_margin",
    "max_family_margin",
    "first_window_start_s",
    "last_window_end_s",
)

CLUSTER_COLUMNS = (
    "rat",
    "session",
    "cluster_index",
    "cluster_id",
    "window_count",
    "template_event_count",
    "template_event_indices",
    "time_start_s",
    "time_end_s",
    "duration_s",
    "median_family_margin",
    "max_family_margin",
    "best_trajectory_model",
    "best_candidate_event_index",
    "best_candidate_null_index",
    "median_n_spikes",
    "median_active_cell_count",
)

BEHAVIOR_LFP_COLUMNS = (
    "feature",
    "feature_available",
    "candidate_class",
    "windows",
    "non_null_windows",
    "mean",
    "median",
    "std",
    "min",
    "max",
)

GATE_COLUMNS = ("gate", "passed", "observed", "criterion", "required_for_overall")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _parse_names(value: str | Iterable[str] | None, default: Sequence[str] = ()) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
        return names or tuple(default)
    names = tuple(str(part).strip() for part in value if str(part).strip())
    return names or tuple(default)


def _read_score_files(score_glob: str | Path) -> pd.DataFrame:
    paths = [Path(path) for path in sorted(glob.glob(str(score_glob), recursive=True))]
    if not paths:
        raise FileNotFoundError(f"no off-SWR score files found for {score_glob!r}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _first_value(frame: pd.DataFrame, column: str) -> object:
    if column not in frame.columns:
        return np.nan
    values = frame[column].dropna()
    return values.iloc[0] if not values.empty else np.nan


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator == 0.0 or not np.isfinite(denominator):
        return np.nan
    return float(numerator) / denominator


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or float(denominator) == 0.0:
        return np.nan
    return float(numerator) / float(denominator)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _first_numeric_from_columns(row: pd.Series | None, columns: Sequence[str]) -> float:
    if row is None:
        return np.nan
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return np.nan


def _finite_or_nan(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else np.nan


def _safe_softmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return np.full(arr.shape, np.nan, dtype=float)
    shifted = arr - np.max(arr)
    weights = np.exp(shifted)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(arr.shape, np.nan, dtype=float)
    return weights / total


def _euclidean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape or not np.all(np.isfinite(left_arr)) or not np.all(np.isfinite(right_arr)):
        return np.nan
    return float(np.linalg.norm(left_arr - right_arr))


def _percentile_rank(value: float, reference: pd.Series) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    if reference.empty or not np.isfinite(value):
        return np.nan
    return float((reference <= float(value)).mean())


def _safe_median(frame: pd.DataFrame, column: str) -> float:
    values = _numeric_series(frame, column).dropna()
    return float(values.median()) if not values.empty else np.nan


def _median_delta(left: pd.DataFrame, right: pd.DataFrame, column: str) -> float:
    left_median = _safe_median(left, column)
    right_median = _safe_median(right, column)
    if not np.isfinite(left_median) or not np.isfinite(right_median):
        return np.nan
    return float(left_median - right_median)


def _success_comparable_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores.copy()
    out = scores.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).eq("success")].copy()
    if "evidence_comparable" in out.columns:
        out = out[out["evidence_comparable"].map(_as_bool)].copy()
    if "model" in out.columns:
        out["model"] = out["model"].astype(str)
    return out


def _score_lookup(scores: pd.DataFrame) -> dict[tuple[str, int, str, int], pd.DataFrame]:
    out: dict[tuple[str, int, str, int], pd.DataFrame] = {}
    if scores.empty or not set(KEY_COLUMNS).issubset(scores.columns):
        return out
    for key, group in _success_comparable_scores(scores).groupby(list(KEY_COLUMNS), sort=False, dropna=False):
        session, event_index, window_role, null_index = key if isinstance(key, tuple) else (key,)
        out[(str(session), int(event_index), str(window_role), int(null_index))] = group.copy()
    return out


def _best_model_row(group: pd.DataFrame, model: object) -> pd.Series | None:
    if group.empty or "model" not in group.columns:
        return None
    rows = group[group["model"].astype(str).eq(str(model))]
    if rows.empty:
        return None
    if "log_evidence" in rows.columns:
        rows = rows.assign(_log_evidence=pd.to_numeric(rows["log_evidence"], errors="coerce"))
        return rows.sort_values("_log_evidence", ascending=False).iloc[0]
    return rows.iloc[0]


def _trajectory_confidence_from_scores(
    group: pd.DataFrame,
    *,
    required_models: Sequence[str],
    trajectory_models: Sequence[str],
) -> float:
    if group.empty or "model" not in group.columns or "log_evidence" not in group.columns:
        return np.nan
    by_model = group[group["model"].astype(str).isin(tuple(required_models))].dropna(subset=["log_evidence"]).drop_duplicates("model", keep="last")
    by_model = by_model.set_index("model")
    if any(model not in by_model.index for model in required_models):
        return np.nan
    logz = [float(by_model.loc[model, "log_evidence"]) for model in required_models]
    probs = _safe_softmax(logz)
    if not np.all(np.isfinite(probs)):
        return np.nan
    probability_by_model = dict(zip(required_models, probs, strict=True))
    return float(sum(probability_by_model.get(model, 0.0) for model in trajectory_models))


def _real_swr_intervals(scores: pd.DataFrame) -> dict[str, np.ndarray]:
    if scores.empty or not {"session", "window_role", "window_start_s", "window_end_s"}.issubset(scores.columns):
        return {}
    real = scores[scores["window_role"].astype(str).eq("real")].copy()
    if real.empty:
        return {}
    real["window_start_s"] = pd.to_numeric(real["window_start_s"], errors="coerce")
    real["window_end_s"] = pd.to_numeric(real["window_end_s"], errors="coerce")
    real = real.dropna(subset=["window_start_s", "window_end_s"]).drop_duplicates(["session", "window_start_s", "window_end_s"])
    out: dict[str, np.ndarray] = {}
    for session, group in real.groupby("session", sort=False):
        out[str(session)] = group[["window_start_s", "window_end_s"]].to_numpy(dtype=float)
    return out


def _distance_to_nearest_interval(start: float, end: float, intervals: np.ndarray) -> tuple[float, bool]:
    if not np.isfinite(start) or not np.isfinite(end) or intervals.size == 0:
        return np.nan, False
    overlaps = bool(np.any((float(start) < intervals[:, 1]) & (float(end) > intervals[:, 0])))
    if overlaps:
        return 0.0, True
    distances = np.minimum(np.abs(float(start) - intervals[:, 1]), np.abs(intervals[:, 0] - float(end)))
    distances = distances[np.isfinite(distances)]
    return (float(distances.min()) if distances.size else np.nan), False


def _decoded_speed(decoded_path_length: float, decoded_start_to_end_distance: float, duration_s: float) -> float:
    distance = decoded_path_length if np.isfinite(decoded_path_length) else decoded_start_to_end_distance
    return _safe_ratio(distance, duration_s)


def _run_or_immobility_state(
    *,
    animal_speed_mean: float,
    animal_speed_median: float,
    animal_speed_max: float,
    run_speed_threshold_cm_s: float,
) -> str:
    speed_for_state = next(
        (value for value in (animal_speed_mean, animal_speed_median, animal_speed_max) if np.isfinite(value)),
        np.nan,
    )
    if not np.isfinite(speed_for_state):
        return "unknown_speed"
    return "run" if speed_for_state >= float(run_speed_threshold_cm_s) else "immobile"


def _format_model_distribution(distribution: pd.DataFrame, window_set: str) -> str:
    if distribution.empty:
        return ""
    subset = distribution[distribution["window_set"].astype(str).eq(window_set)].copy()
    if subset.empty:
        return ""
    subset = subset.sort_values(["windows", "best_trajectory_model"], ascending=[False, True])
    return "; ".join(
        f"{row.best_trajectory_model}={int(row.windows)} ({float(row.fraction):.3f})"
        for row in subset.itertuples(index=False)
    )


def _format_model_distribution_from_frame(frame: pd.DataFrame) -> str:
    if frame.empty or "best_trajectory_model" not in frame.columns:
        return ""
    models = frame["best_trajectory_model"].dropna().astype(str)
    models = models[models.ne("")]
    if models.empty:
        return ""
    counts = models.value_counts(sort=True)
    total = float(counts.sum())
    return "; ".join(f"{model}={int(count)} ({float(count) / total:.3f})" for model, count in counts.items())


def _candidate_tier_for_margin(
    margin: float,
    thresholds: Sequence[tuple[str, float]] = DEFAULT_CANDIDATE_TIER_THRESHOLDS,
) -> str:
    if not np.isfinite(margin):
        return ""
    tier = ""
    for label, threshold in thresholds:
        if float(margin) >= float(threshold):
            tier = str(label)
    return tier


def _tier_candidate_mask(frame: pd.DataFrame, threshold: float) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    return _numeric_series(frame, "trajectory_family_margin") >= float(threshold)


def _window_metadata(scores: pd.DataFrame, *, optional_columns: Sequence[str]) -> pd.DataFrame:
    if scores.empty:
        return _empty_frame(KEY_COLUMNS)
    metadata_columns = (
        "off_swr",
        "template_event_index",
        "matched_null_rank",
        "real_event_start_s",
        "real_event_end_s",
        "real_event_duration_s",
        "real_active_cell_count",
        "null_n_spikes",
        "null_active_cell_count",
        "active_cell_count_delta",
        "restrict_to_run_times",
        "window_index",
        *optional_columns,
    )
    present = [column for column in metadata_columns if column in scores.columns and column not in KEY_COLUMNS]
    rows: list[dict[str, object]] = []
    for key, group in scores.groupby(list(KEY_COLUMNS), sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(KEY_COLUMNS, key_tuple, strict=True)}
        for column in present:
            row[column] = _first_value(group, column)
        rows.append(row)
    return pd.DataFrame(rows)


def off_swr_trajectory_decisions(
    scores: pd.DataFrame,
    *,
    comparison_scope: str = "full-core",
    required_models: tuple[str, ...] | None = None,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    behavior_lfp_columns: Sequence[str] = DEFAULT_BEHAVIOR_LFP_COLUMNS,
) -> pd.DataFrame:
    """Return one classified off-SWR discovery row per matched-null window."""

    if scores.empty:
        return _empty_frame(DECISION_OUTPUT_COLUMNS)

    decisions = matched_null_family_margin_decisions(
        scores,
        comparison_scope=comparison_scope,
        required_models=required_models,
        margin_threshold=margin_threshold,
    )
    if decisions.empty:
        return _empty_frame(DECISION_OUTPUT_COLUMNS)

    metadata = _window_metadata(scores, optional_columns=behavior_lfp_columns)
    metadata = metadata[[column for column in metadata.columns if column in KEY_COLUMNS or column not in decisions.columns]]
    decisions = decisions.merge(metadata, on=list(KEY_COLUMNS), how="left")
    decisions = decisions[decisions["window_role"].astype(str).eq("matched_null")].copy()
    if decisions.empty:
        return _empty_frame(DECISION_OUTPUT_COLUMNS)

    decisions["session"] = decisions["session"].astype(str)
    decisions["rat"] = decisions["session"].map(_rat_from_session)
    decisions["off_swr"] = decisions["off_swr"].map(_as_bool) if "off_swr" in decisions.columns else True
    decisions["passes_known_swr_exclusion"] = decisions["off_swr"].map(_as_bool)
    decisions["required_models_complete"] = decisions["required_models_complete"].map(_as_bool)
    decisions["trajectory_confident_claim"] = decisions["trajectory_confident_claim"].map(_as_bool)
    decisions["nontrajectory_confident_claim"] = decisions["nontrajectory_confident_claim"].map(_as_bool)

    def classify(row: pd.Series) -> str:
        if not bool(row["passes_known_swr_exclusion"]):
            return EXCLUDED_SWR_OVERLAP_CLASS
        if not bool(row["required_models_complete"]):
            return INCOMPLETE_CLASS
        if bool(row["trajectory_confident_claim"]):
            return TRAJECTORY_CANDIDATE_CLASS
        if bool(row["nontrajectory_confident_claim"]):
            return STATIC_NONTRAJECTORY_CLASS
        return AMBIGUOUS_CLASS

    decisions["candidate_class"] = decisions.apply(classify, axis=1)
    decisions["is_trajectory_family_candidate"] = decisions["candidate_class"].eq(TRAJECTORY_CANDIDATE_CLASS)
    return decisions.sort_values(["session", "window_start_s", "event_index", "null_index"], na_position="last").reset_index(drop=True)


def off_swr_trajectory_candidates(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return _empty_frame(DECISION_OUTPUT_COLUMNS)
    candidates = decisions[decisions["is_trajectory_family_candidate"].map(_as_bool)].copy()
    return candidates.reset_index(drop=True)


def off_swr_candidate_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return _empty_frame(SUMMARY_COLUMNS)

    off_swr = decisions[decisions["passes_known_swr_exclusion"].map(_as_bool)].copy()
    candidates = off_swr[off_swr["is_trajectory_family_candidate"].map(_as_bool)].copy()
    margins = _numeric_series(off_swr, "trajectory_minus_nontrajectory_log_evidence")
    candidate_margins = _numeric_series(candidates, "trajectory_minus_nontrajectory_log_evidence")
    row = {
        "comparison_scope": str(_first_value(decisions, "comparison_scope")),
        "windows": int(len(decisions)),
        "off_swr_windows": int(len(off_swr)),
        "required_complete_windows": int(decisions["required_models_complete"].map(_as_bool).sum()),
        "trajectory_family_candidates": int(len(candidates)),
        "static_nontrajectory_windows": int(decisions["candidate_class"].eq(STATIC_NONTRAJECTORY_CLASS).sum()),
        "ambiguous_windows": int(decisions["candidate_class"].eq(AMBIGUOUS_CLASS).sum()),
        "incomplete_windows": int(decisions["candidate_class"].eq(INCOMPLETE_CLASS).sum()),
        "excluded_known_swr_overlap_windows": int(decisions["candidate_class"].eq(EXCLUDED_SWR_OVERLAP_CLASS).sum()),
        "candidate_fraction_of_off_swr": _safe_fraction(len(candidates), len(off_swr)),
        "candidate_sessions": int(candidates["session"].nunique()) if not candidates.empty else 0,
        "candidate_rats": int(candidates["rat"].nunique()) if not candidates.empty else 0,
        "mean_family_margin": float(margins.mean()) if not margins.dropna().empty else np.nan,
        "median_family_margin": float(margins.median()) if not margins.dropna().empty else np.nan,
        "mean_candidate_family_margin": float(candidate_margins.mean()) if not candidate_margins.dropna().empty else np.nan,
        "median_candidate_family_margin": float(candidate_margins.median()) if not candidate_margins.dropna().empty else np.nan,
        "max_candidate_family_margin": float(candidate_margins.max()) if not candidate_margins.dropna().empty else np.nan,
        "margin_threshold": float(_first_value(decisions, "margin_threshold")),
    }
    return pd.DataFrame([row], columns=list(SUMMARY_COLUMNS))


def off_swr_group_summary(decisions: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if decisions.empty:
        return _empty_frame(GROUP_SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    for key, group in decisions.groupby(list(group_cols), sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        off_swr = group[group["passes_known_swr_exclusion"].map(_as_bool)]
        candidates = off_swr[off_swr["is_trajectory_family_candidate"].map(_as_bool)]
        margins = _numeric_series(off_swr, "trajectory_minus_nontrajectory_log_evidence")
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "comparison_scope": str(_first_value(group, "comparison_scope")),
                "rat": str(_first_value(group, "rat")),
                "session": str(_first_value(group, "session")),
                "windows": int(len(group)),
                "off_swr_windows": int(len(off_swr)),
                "trajectory_family_candidates": int(len(candidates)),
                "static_nontrajectory_windows": int(group["candidate_class"].eq(STATIC_NONTRAJECTORY_CLASS).sum()),
                "ambiguous_windows": int(group["candidate_class"].eq(AMBIGUOUS_CLASS).sum()),
                "candidate_fraction_of_off_swr": _safe_fraction(len(candidates), len(off_swr)),
                "median_family_margin": float(margins.median()) if not margins.dropna().empty else np.nan,
                "max_family_margin": float(margins.max()) if not margins.dropna().empty else np.nan,
                "first_window_start_s": float(_numeric_series(group, "window_start_s").min()),
                "last_window_end_s": float(_numeric_series(group, "window_end_s").max()),
            }
        )
        rows.append(row)
    columns = [column for column in GROUP_SUMMARY_COLUMNS if column in {"comparison_scope", *group_cols} or column not in group_cols]
    return pd.DataFrame(rows)[columns]


def cluster_off_swr_candidates(candidates: pd.DataFrame, *, cluster_gap_s: float = 0.5) -> pd.DataFrame:
    if candidates.empty:
        return _empty_frame(CLUSTER_COLUMNS)

    rows: list[dict[str, object]] = []

    def emit_cluster(session: str, cluster_index: int, cluster_rows: list[pd.Series]) -> None:
        cluster = pd.DataFrame(cluster_rows)
        starts = _numeric_series(cluster, "window_start_s")
        ends = _numeric_series(cluster, "window_end_s")
        margins = _numeric_series(cluster, "trajectory_minus_nontrajectory_log_evidence")
        best = cluster.assign(_margin=margins).sort_values(["_margin", "window_start_s"], ascending=[False, True]).iloc[0]
        time_start = float(starts.min()) if not starts.dropna().empty else np.nan
        time_end = float(ends.max()) if not ends.dropna().empty else np.nan
        template_events = tuple(sorted(set(pd.to_numeric(cluster["event_index"], errors="coerce").dropna().astype(int))))
        rows.append(
            {
                "rat": str(best["rat"]),
                "session": session,
                "cluster_index": int(cluster_index),
                "cluster_id": f"{session.replace('/', '_')}_off_swr_cluster_{cluster_index:04d}",
                "window_count": int(len(cluster)),
                "template_event_count": int(len(template_events)),
                "template_event_indices": " ".join(str(event) for event in template_events),
                "time_start_s": time_start,
                "time_end_s": time_end,
                "duration_s": float(time_end - time_start) if np.isfinite(time_start) and np.isfinite(time_end) else np.nan,
                "median_family_margin": float(margins.median()) if not margins.dropna().empty else np.nan,
                "max_family_margin": float(margins.max()) if not margins.dropna().empty else np.nan,
                "best_trajectory_model": str(best["best_trajectory_model"]),
                "best_candidate_event_index": int(best["event_index"]),
                "best_candidate_null_index": int(best["null_index"]),
                "median_n_spikes": float(_numeric_series(cluster, "n_spikes").median()),
                "median_active_cell_count": float(_numeric_series(cluster, "active_cell_count").median()),
            }
        )

    for session, group in candidates.groupby("session", sort=True):
        group = group.sort_values(["window_start_s", "window_end_s", "event_index", "null_index"], na_position="last")
        cluster_rows: list[pd.Series] = []
        cluster_index = 0
        previous_end = np.nan
        for _, row in group.iterrows():
            start = pd.to_numeric(row.get("window_start_s"), errors="coerce")
            end = pd.to_numeric(row.get("window_end_s"), errors="coerce")
            should_start_new = bool(
                cluster_rows
                and (
                    not np.isfinite(float(start))
                    or not np.isfinite(float(previous_end))
                    or float(start) - float(previous_end) > float(cluster_gap_s)
                )
            )
            if should_start_new:
                emit_cluster(str(session), cluster_index, cluster_rows)
                cluster_index += 1
                cluster_rows = []
            cluster_rows.append(row)
            if np.isfinite(float(end)):
                previous_end = max(float(previous_end), float(end)) if np.isfinite(float(previous_end)) else float(end)
        if cluster_rows:
            emit_cluster(str(session), cluster_index, cluster_rows)
    return pd.DataFrame(rows, columns=list(CLUSTER_COLUMNS))


def _assign_candidate_cluster_ids(candidates: pd.DataFrame, *, cluster_gap_s: float = 0.5) -> pd.Series:
    if candidates.empty:
        return pd.Series(dtype=object)
    ids = pd.Series("", index=candidates.index, dtype=object)
    for session, group in candidates.groupby("session", sort=True):
        ordered = group.sort_values(["window_start_s", "window_end_s", "event_index", "null_index"], na_position="last")
        cluster_index = 0
        previous_end = np.nan
        cluster_started = False
        for index, row in ordered.iterrows():
            start = _finite_or_nan(row.get("window_start_s"))
            end = _finite_or_nan(row.get("window_end_s"))
            if cluster_started and (
                not np.isfinite(start)
                or not np.isfinite(previous_end)
                or start - previous_end > float(cluster_gap_s)
            ):
                cluster_index += 1
            ids.loc[index] = f"{str(session).replace('/', '_')}_off_swr_cluster_{cluster_index:04d}"
            cluster_started = True
            if np.isfinite(end):
                previous_end = max(float(previous_end), float(end)) if np.isfinite(float(previous_end)) else float(end)
    return ids


def off_swr_candidate_table(
    decisions: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    required_models: Sequence[str] = DEFAULT_OFF_SWR_DISCOVERY_MODELS,
    trajectory_models: Sequence[str] = FULL_CORE_REQUIRED_MODELS[1:],
    cluster_gap_s: float = 0.5,
    run_speed_threshold_cm_s: float = 5.0,
) -> pd.DataFrame:
    """Return ranked, phenotyped off-SWR trajectory-family candidate windows."""

    if decisions.empty:
        return _empty_frame(CANDIDATE_TABLE_COLUMNS)

    candidates = off_swr_trajectory_candidates(decisions)
    if candidates.empty:
        return _empty_frame(CANDIDATE_TABLE_COLUMNS)

    required = tuple(str(model) for model in required_models)
    trajectory = tuple(str(model) for model in trajectory_models if str(model) in set(required))
    lookup = _score_lookup(scores)
    swr_intervals = _real_swr_intervals(scores)
    off_swr_reference = decisions[decisions["passes_known_swr_exclusion"].map(_as_bool)].copy()
    spike_reference = _numeric_series(off_swr_reference, "n_spikes")
    active_reference = _numeric_series(off_swr_reference, "active_cell_count")

    rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        key = (
            str(row["session"]),
            int(row["event_index"]),
            str(row["window_role"]),
            int(row["null_index"]),
        )
        group = lookup.get(key, pd.DataFrame())
        best_trajectory = _best_model_row(group, row.get("best_trajectory_model"))

        start = _finite_or_nan(row.get("window_start_s"))
        end = _finite_or_nan(row.get("window_end_s"))
        duration = _finite_or_nan(row.get("window_duration_s"))
        distance_to_swr, overlaps_swr = _distance_to_nearest_interval(start, end, swr_intervals.get(str(row["session"]), np.empty((0, 2))))
        overlaps_known_swr = bool(overlaps_swr or not _as_bool(row.get("passes_known_swr_exclusion")))

        margin = _finite_or_nan(row.get("trajectory_minus_nontrajectory_log_evidence"))
        confidence = _trajectory_confidence_from_scores(group, required_models=required, trajectory_models=trajectory)
        entropy = _first_numeric_from_columns(best_trajectory, TRIAGE_ENTROPY_COLUMNS)
        animal_speed_mean = _first_numeric_from_columns(best_trajectory, TRIAGE_MEAN_SPEED_COLUMNS)
        animal_speed_median = _first_numeric_from_columns(best_trajectory, TRIAGE_MEDIAN_SPEED_COLUMNS)
        animal_speed_max = _first_numeric_from_columns(best_trajectory, TRIAGE_MAX_SPEED_COLUMNS)
        position_sample_count = _first_numeric_from_columns(best_trajectory, ("position_sample_count",))
        run_state = _run_or_immobility_state(
            animal_speed_mean=animal_speed_mean,
            animal_speed_median=animal_speed_median,
            animal_speed_max=animal_speed_max,
            run_speed_threshold_cm_s=run_speed_threshold_cm_s,
        )

        endpoint_x = _first_numeric_from_columns(best_trajectory, TRIAGE_ENDPOINT_X_COLUMNS)
        endpoint_y = _first_numeric_from_columns(best_trajectory, TRIAGE_ENDPOINT_Y_COLUMNS)
        start_x = _first_numeric_from_columns(best_trajectory, TRIAGE_START_X_COLUMNS)
        start_y = _first_numeric_from_columns(best_trajectory, TRIAGE_START_Y_COLUMNS)
        animal_x = _first_numeric_from_columns(best_trajectory, TRIAGE_ANIMAL_X_COLUMNS)
        animal_y = _first_numeric_from_columns(best_trajectory, TRIAGE_ANIMAL_Y_COLUMNS)
        decoded_start_to_end = _euclidean_distance((start_x, start_y), (endpoint_x, endpoint_y))
        decoded_endpoint_distance = _euclidean_distance((animal_x, animal_y), (endpoint_x, endpoint_y))
        decoded_path_length = _first_numeric_from_columns(best_trajectory, TRIAGE_PATH_LENGTH_COLUMNS)
        decoded_speed = _decoded_speed(decoded_path_length, decoded_start_to_end, duration)

        n_spikes = _finite_or_nan(row.get("n_spikes"))
        active_cell_count = _finite_or_nan(row.get("active_cell_count"))
        spike_percentile = _percentile_rank(n_spikes, spike_reference)
        active_percentile = _percentile_rank(active_cell_count, active_reference)
        speed_percentile = 1.0 if run_state == "run" else np.nan
        ordinary_components = [value for value in (spike_percentile, active_percentile, speed_percentile) if np.isfinite(value)]
        ordinary_score = float(np.mean(ordinary_components)) if ordinary_components else np.nan
        low_information = bool(
            (np.isfinite(spike_percentile) and spike_percentile <= 0.10)
            or (np.isfinite(active_percentile) and active_percentile <= 0.10)
        )
        if low_information:
            specificity_label = LOW_INFORMATION_LABEL
        elif np.isfinite(ordinary_score) and ordinary_score >= 0.75:
            specificity_label = MOVEMENT_SPIKING_LIKE_LABEL
        else:
            specificity_label = INTERESTING_CANDIDATE_LABEL

        margin_component = float(np.log1p(max(margin, 0.0))) if np.isfinite(margin) else 0.0
        confidence_component = confidence if np.isfinite(confidence) else 0.0
        ordinary_penalty = ordinary_score if np.isfinite(ordinary_score) else 0.0
        entropy_penalty = min(entropy / 10.0, 1.0) if np.isfinite(entropy) else 0.0
        priority = margin_component + confidence_component - ordinary_penalty - entropy_penalty

        rows.append(
            {
                "candidate_specificity_label": specificity_label,
                "candidate_priority_score": float(priority),
                "ordinary_movement_spiking_score": ordinary_score,
                "session": str(row["session"]),
                "rat": str(row["rat"]),
                "event_index": int(row["event_index"]),
                "null_index": int(row["null_index"]),
                "window_start_s": start,
                "window_end_s": end,
                "duration_s": duration,
                "n_spikes": n_spikes,
                "active_cell_count": active_cell_count,
                "trajectory_family_margin": margin,
                "candidate_tier": _candidate_tier_for_margin(margin),
                "best_trajectory_model": str(row.get("best_trajectory_model", "")),
                "trajectory_confidence": confidence,
                "trajectory_posterior_entropy": entropy,
                "distance_to_nearest_swr_s": distance_to_swr,
                "overlaps_known_swr": overlaps_known_swr,
                "animal_speed_mean": animal_speed_mean,
                "animal_speed_median": animal_speed_median,
                "animal_speed_max": animal_speed_max,
                "position_sample_count": position_sample_count,
                "run_or_immobility_state": run_state,
                "decoded_path_length": decoded_path_length,
                "decoded_speed": decoded_speed,
                "decoded_endpoint_distance": decoded_endpoint_distance,
                "decoded_start_to_end_distance": decoded_start_to_end,
                "candidate_cluster_id": "",
                "comparison_scope": str(row.get("comparison_scope", "")),
                "margin_threshold": _finite_or_nan(row.get("margin_threshold")),
                "best_nontrajectory_model": str(row.get("best_nontrajectory_model", "")),
                "trajectory_margin_per_spike": _finite_or_nan(row.get("trajectory_minus_nontrajectory_log_evidence_per_spike")),
                "trajectory_margin_per_time_bin": _finite_or_nan(row.get("trajectory_minus_nontrajectory_log_evidence_per_time_bin")),
                "matched_null_rank": _finite_or_nan(row.get("matched_null_rank")),
                "template_event_index": _finite_or_nan(row.get("template_event_index")),
            }
        )

    table = pd.DataFrame(rows)
    table["candidate_cluster_id"] = _assign_candidate_cluster_ids(table, cluster_gap_s=cluster_gap_s)
    table = table.sort_values(
        ["candidate_priority_score", "trajectory_family_margin", "window_start_s"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    table.insert(0, "candidate_rank", np.arange(1, len(table) + 1, dtype=int))
    return table[[column for column in CANDIDATE_TABLE_COLUMNS if column in table.columns]]


def off_swr_candidate_cluster_table(candidate_table: pd.DataFrame) -> pd.DataFrame:
    if candidate_table.empty:
        return _empty_frame(CANDIDATE_CLUSTER_TABLE_COLUMNS)
    rows: list[dict[str, object]] = []
    for cluster_id, group in candidate_table.groupby("candidate_cluster_id", sort=True):
        starts = _numeric_series(group, "window_start_s")
        ends = _numeric_series(group, "window_end_s")
        best = group.sort_values(["candidate_priority_score", "trajectory_family_margin"], ascending=[False, False]).iloc[0]
        time_start = float(starts.min()) if not starts.dropna().empty else np.nan
        time_end = float(ends.max()) if not ends.dropna().empty else np.nan
        template_events = tuple(sorted(set(pd.to_numeric(group["event_index"], errors="coerce").dropna().astype(int))))
        rows.append(
            {
                "rat": str(best["rat"]),
                "session": str(best["session"]),
                "candidate_cluster_id": str(cluster_id),
                "cluster_index": int(str(cluster_id).rsplit("_", 1)[-1]) if str(cluster_id).rsplit("_", 1)[-1].isdigit() else len(rows),
                "window_count": int(len(group)),
                "template_event_count": int(len(template_events)),
                "template_event_indices": " ".join(str(event) for event in template_events),
                "time_start_s": time_start,
                "time_end_s": time_end,
                "duration_s": float(time_end - time_start) if np.isfinite(time_start) and np.isfinite(time_end) else np.nan,
                "median_family_margin": _safe_median(group, "trajectory_family_margin"),
                "max_family_margin": float(_numeric_series(group, "trajectory_family_margin").max()),
                "median_trajectory_confidence": _safe_median(group, "trajectory_confidence"),
                "median_priority_score": _safe_median(group, "candidate_priority_score"),
                "movement_spiking_like_windows": int(group["candidate_specificity_label"].astype(str).eq(MOVEMENT_SPIKING_LIKE_LABEL).sum()),
                "interesting_candidate_windows": int(group["candidate_specificity_label"].astype(str).eq(INTERESTING_CANDIDATE_LABEL).sum()),
                "best_trajectory_model": str(best["best_trajectory_model"]),
                "best_candidate_event_index": int(best["event_index"]),
                "best_candidate_null_index": int(best["null_index"]),
                "median_n_spikes": _safe_median(group, "n_spikes"),
                "median_active_cell_count": _safe_median(group, "active_cell_count"),
                "median_animal_speed_mean": _safe_median(group, "animal_speed_mean"),
                "min_distance_to_nearest_swr_s": float(_numeric_series(group, "distance_to_nearest_swr_s").min()),
            }
        )
    return pd.DataFrame(rows, columns=list(CANDIDATE_CLUSTER_TABLE_COLUMNS))


def off_swr_candidate_group_summary(candidate_table: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if candidate_table.empty:
        return _empty_frame(CANDIDATE_GROUP_SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    for key, group in candidate_table.groupby(list(group_cols), sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "comparison_scope": str(_first_value(group, "comparison_scope")),
                "rat": str(_first_value(group, "rat")),
                "session": str(_first_value(group, "session")),
                "candidate_windows": int(len(group)),
                "candidate_clusters": int(group["candidate_cluster_id"].nunique()),
                "interesting_candidate_windows": int(group["candidate_specificity_label"].astype(str).eq(INTERESTING_CANDIDATE_LABEL).sum()),
                "movement_spiking_like_windows": int(group["candidate_specificity_label"].astype(str).eq(MOVEMENT_SPIKING_LIKE_LABEL).sum()),
                "low_information_candidate_windows": int(group["candidate_specificity_label"].astype(str).eq(LOW_INFORMATION_LABEL).sum()),
                "median_candidate_priority_score": _safe_median(group, "candidate_priority_score"),
                "median_family_margin": _safe_median(group, "trajectory_family_margin"),
                "median_trajectory_confidence": _safe_median(group, "trajectory_confidence"),
                "median_distance_to_nearest_swr_s": _safe_median(group, "distance_to_nearest_swr_s"),
                "median_n_spikes": _safe_median(group, "n_spikes"),
                "median_active_cell_count": _safe_median(group, "active_cell_count"),
                "median_animal_speed_mean": _safe_median(group, "animal_speed_mean"),
            }
        )
        rows.append(row)
    columns = [column for column in CANDIDATE_GROUP_SUMMARY_COLUMNS if column in {"comparison_scope", *group_cols} or column not in group_cols]
    return pd.DataFrame(rows)[columns]


def off_swr_candidate_vs_swr_window_table(
    candidate_table: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    comparison_scope: str = "full-core",
    required_models: tuple[str, ...] | None = None,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    run_speed_threshold_cm_s: float = 5.0,
) -> pd.DataFrame:
    if candidate_table.empty:
        return _empty_frame(CANDIDATE_VS_SWR_WINDOW_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, row in candidate_table.iterrows():
        decoded_speed = _finite_or_nan(row.get("decoded_speed"))
        if not np.isfinite(decoded_speed):
            decoded_speed = _decoded_speed(
                _finite_or_nan(row.get("decoded_path_length")),
                _finite_or_nan(row.get("decoded_start_to_end_distance")),
                _finite_or_nan(row.get("duration_s")),
            )
        rows.append(
            {
                "comparison_scope": str(row.get("comparison_scope", "")),
                "window_set": "off_swr_candidate",
                "session": str(row.get("session", "")),
                "rat": str(row.get("rat", "")),
                "event_index": _finite_or_nan(row.get("event_index")),
                "window_role": "matched_null",
                "null_index": _finite_or_nan(row.get("null_index")),
                "candidate_rank": _finite_or_nan(row.get("candidate_rank")),
                "candidate_specificity_label": str(row.get("candidate_specificity_label", "")),
                "candidate_cluster_id": str(row.get("candidate_cluster_id", "")),
                "window_start_s": _finite_or_nan(row.get("window_start_s")),
                "window_end_s": _finite_or_nan(row.get("window_end_s")),
                "duration_s": _finite_or_nan(row.get("duration_s")),
                "n_spikes": _finite_or_nan(row.get("n_spikes")),
                "active_cell_count": _finite_or_nan(row.get("active_cell_count")),
                "trajectory_family_margin": _finite_or_nan(row.get("trajectory_family_margin")),
                "best_trajectory_model": str(row.get("best_trajectory_model", "")),
                "trajectory_confidence": _finite_or_nan(row.get("trajectory_confidence")),
                "trajectory_posterior_entropy": _finite_or_nan(row.get("trajectory_posterior_entropy")),
                "decoded_path_length": _finite_or_nan(row.get("decoded_path_length")),
                "decoded_speed": decoded_speed,
                "decoded_endpoint_distance": _finite_or_nan(row.get("decoded_endpoint_distance")),
                "decoded_start_to_end_distance": _finite_or_nan(row.get("decoded_start_to_end_distance")),
                "animal_speed_mean": _finite_or_nan(row.get("animal_speed_mean")),
                "animal_speed_median": _finite_or_nan(row.get("animal_speed_median")),
                "animal_speed_max": _finite_or_nan(row.get("animal_speed_max")),
                "position_sample_count": _finite_or_nan(row.get("position_sample_count")),
                "run_or_immobility_state": str(row.get("run_or_immobility_state", "")),
                "distance_to_nearest_swr_s": _finite_or_nan(row.get("distance_to_nearest_swr_s")),
                "overlaps_known_swr": bool(row.get("overlaps_known_swr", False)),
            }
        )

    swr_decisions = matched_null_family_margin_decisions(
        scores,
        comparison_scope=comparison_scope,
        required_models=required_models,
        margin_threshold=margin_threshold,
    )
    swr_decisions = swr_decisions[swr_decisions["window_role"].astype(str).eq("real")].copy() if not swr_decisions.empty else pd.DataFrame()
    if not swr_decisions.empty:
        lookup = _score_lookup(scores)
        required = tuple(required_models or DEFAULT_OFF_SWR_DISCOVERY_MODELS)
        trajectory = tuple(model for model in FULL_CORE_REQUIRED_MODELS[1:] if model in set(required))
        swr_intervals = _real_swr_intervals(scores)
        for _, row in swr_decisions.iterrows():
            session = str(row["session"])
            event_index = int(row["event_index"])
            window_role = str(row["window_role"])
            null_index = int(row["null_index"])
            group = lookup.get((session, event_index, window_role, null_index), pd.DataFrame())
            best_trajectory = _best_model_row(group, row.get("best_trajectory_model"))
            start = _finite_or_nan(row.get("window_start_s"))
            end = _finite_or_nan(row.get("window_end_s"))
            duration = _finite_or_nan(row.get("window_duration_s"))
            distance_to_swr, overlaps_swr = _distance_to_nearest_interval(
                start,
                end,
                swr_intervals.get(session, np.empty((0, 2))),
            )
            animal_speed_mean = _first_numeric_from_columns(best_trajectory, TRIAGE_MEAN_SPEED_COLUMNS)
            animal_speed_median = _first_numeric_from_columns(best_trajectory, TRIAGE_MEDIAN_SPEED_COLUMNS)
            animal_speed_max = _first_numeric_from_columns(best_trajectory, TRIAGE_MAX_SPEED_COLUMNS)
            position_sample_count = _first_numeric_from_columns(best_trajectory, ("position_sample_count",))
            endpoint_x = _first_numeric_from_columns(best_trajectory, TRIAGE_ENDPOINT_X_COLUMNS)
            endpoint_y = _first_numeric_from_columns(best_trajectory, TRIAGE_ENDPOINT_Y_COLUMNS)
            start_x = _first_numeric_from_columns(best_trajectory, TRIAGE_START_X_COLUMNS)
            start_y = _first_numeric_from_columns(best_trajectory, TRIAGE_START_Y_COLUMNS)
            animal_x = _first_numeric_from_columns(best_trajectory, TRIAGE_ANIMAL_X_COLUMNS)
            animal_y = _first_numeric_from_columns(best_trajectory, TRIAGE_ANIMAL_Y_COLUMNS)
            decoded_start_to_end = _euclidean_distance((start_x, start_y), (endpoint_x, endpoint_y))
            decoded_endpoint_distance = _euclidean_distance((animal_x, animal_y), (endpoint_x, endpoint_y))
            decoded_path_length = _first_numeric_from_columns(best_trajectory, TRIAGE_PATH_LENGTH_COLUMNS)
            rows.append(
                {
                    "comparison_scope": str(row.get("comparison_scope", comparison_scope)),
                    "window_set": "swr_replay",
                    "session": session,
                    "rat": str(row.get("rat", _rat_from_session(session))),
                    "event_index": event_index,
                    "window_role": window_role,
                    "null_index": null_index,
                    "candidate_rank": np.nan,
                    "candidate_specificity_label": "swr_replay_reference",
                    "candidate_cluster_id": "",
                    "window_start_s": start,
                    "window_end_s": end,
                    "duration_s": duration,
                    "n_spikes": _finite_or_nan(row.get("n_spikes")),
                    "active_cell_count": _finite_or_nan(row.get("active_cell_count")),
                    "trajectory_family_margin": _finite_or_nan(row.get("trajectory_minus_nontrajectory_log_evidence")),
                    "best_trajectory_model": str(row.get("best_trajectory_model", "")),
                    "trajectory_confidence": _trajectory_confidence_from_scores(group, required_models=required, trajectory_models=trajectory),
                    "trajectory_posterior_entropy": _first_numeric_from_columns(best_trajectory, TRIAGE_ENTROPY_COLUMNS),
                    "decoded_path_length": decoded_path_length,
                    "decoded_speed": _decoded_speed(decoded_path_length, decoded_start_to_end, duration),
                    "decoded_endpoint_distance": decoded_endpoint_distance,
                    "decoded_start_to_end_distance": decoded_start_to_end,
                    "animal_speed_mean": animal_speed_mean,
                    "animal_speed_median": animal_speed_median,
                    "animal_speed_max": animal_speed_max,
                    "position_sample_count": position_sample_count,
                    "run_or_immobility_state": _run_or_immobility_state(
                        animal_speed_mean=animal_speed_mean,
                        animal_speed_median=animal_speed_median,
                        animal_speed_max=animal_speed_max,
                        run_speed_threshold_cm_s=run_speed_threshold_cm_s,
                    ),
                    "distance_to_nearest_swr_s": distance_to_swr,
                    "overlaps_known_swr": bool(overlaps_swr),
                }
            )
    return pd.DataFrame(rows, columns=list(CANDIDATE_VS_SWR_WINDOW_COLUMNS))


def off_swr_candidate_vs_swr_model_distribution(window_table: pd.DataFrame) -> pd.DataFrame:
    if window_table.empty:
        return _empty_frame(CANDIDATE_VS_SWR_MODEL_DISTRIBUTION_COLUMNS)
    working = window_table.copy()
    working["best_trajectory_model"] = working["best_trajectory_model"].astype(str)
    working = working[working["best_trajectory_model"].ne("")]
    if working.empty:
        return _empty_frame(CANDIDATE_VS_SWR_MODEL_DISTRIBUTION_COLUMNS)
    counts = (
        working.groupby(["comparison_scope", "window_set", "best_trajectory_model"], sort=True)
        .size()
        .rename("windows")
        .reset_index()
    )
    denominators = counts.groupby(["comparison_scope", "window_set"])["windows"].transform("sum")
    counts["fraction"] = counts["windows"].astype(float) / denominators.astype(float)
    return counts[list(CANDIDATE_VS_SWR_MODEL_DISTRIBUTION_COLUMNS)]


def _interpret_off_swr_vs_swr(summary_row: dict[str, object]) -> tuple[str, bool]:
    candidate_windows = int(summary_row.get("off_swr_candidate_windows", 0) or 0)
    swr_windows = int(summary_row.get("swr_reference_windows", 0) or 0)
    if candidate_windows == 0:
        return "no_off_swr_candidates", False
    if swr_windows == 0:
        return "insufficient_swr_reference", False

    movement_fraction = _finite_or_nan(summary_row.get("candidate_fraction_movement_spiking_like"))
    run_fraction = _finite_or_nan(summary_row.get("candidate_fraction_run"))
    movement_danger = (np.isfinite(movement_fraction) and movement_fraction > 0.5) or (
        np.isfinite(run_fraction) and run_fraction > 0.5
    )
    if movement_danger:
        return "C_mostly_movement_behavioral_decoding_windows", True

    candidate_margin = _finite_or_nan(summary_row.get("candidate_median_family_margin"))
    swr_margin = _finite_or_nan(summary_row.get("swr_median_family_margin"))
    candidate_confidence = _finite_or_nan(summary_row.get("candidate_median_trajectory_confidence"))
    swr_confidence = _finite_or_nan(summary_row.get("swr_median_trajectory_confidence"))
    margin_swr_like = np.isfinite(candidate_margin) and np.isfinite(swr_margin) and candidate_margin >= 0.8 * swr_margin
    confidence_swr_like = (
        np.isfinite(candidate_confidence)
        and np.isfinite(swr_confidence)
        and candidate_confidence >= 0.8 * swr_confidence
    )
    if margin_swr_like and confidence_swr_like:
        return "A_swr_like_strength", False
    if np.isfinite(candidate_margin) and candidate_margin > 0.0:
        return "B_weaker_but_directionally_similar_tail", False
    return "weak_or_ambiguous_off_swr_tail", False


def off_swr_candidate_vs_swr_summary(
    candidate_table: pd.DataFrame,
    window_table: pd.DataFrame,
    model_distribution: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_table.empty:
        return _empty_frame(CANDIDATE_VS_SWR_COLUMNS)

    candidates = window_table[window_table["window_set"].astype(str).eq("off_swr_candidate")].copy() if not window_table.empty else pd.DataFrame()
    swr_reference = window_table[window_table["window_set"].astype(str).eq("swr_replay")].copy() if not window_table.empty else pd.DataFrame()
    movement_like = int(candidate_table["candidate_specificity_label"].astype(str).eq(MOVEMENT_SPIKING_LIKE_LABEL).sum())
    interesting = int(candidate_table["candidate_specificity_label"].astype(str).eq(INTERESTING_CANDIDATE_LABEL).sum())
    run_candidates = int(candidates["run_or_immobility_state"].astype(str).eq("run").sum()) if not candidates.empty else 0
    row = {
        "comparison_scope": str(_first_value(candidate_table, "comparison_scope")),
        "off_swr_candidate_windows": int(len(candidates)),
        "swr_reference_windows": int(len(swr_reference)),
        "candidate_median_family_margin": _safe_median(candidate_table, "trajectory_family_margin"),
        "swr_median_family_margin": _safe_median(swr_reference, "trajectory_family_margin"),
        "candidate_minus_swr_median_family_margin": _median_delta(candidate_table, swr_reference, "trajectory_family_margin"),
        "candidate_median_trajectory_confidence": _safe_median(candidate_table, "trajectory_confidence"),
        "swr_median_trajectory_confidence": _safe_median(swr_reference, "trajectory_confidence"),
        "candidate_minus_swr_median_trajectory_confidence": _median_delta(candidate_table, swr_reference, "trajectory_confidence"),
        "candidate_median_n_spikes": _safe_median(candidate_table, "n_spikes"),
        "swr_median_n_spikes": _safe_median(swr_reference, "n_spikes"),
        "candidate_minus_swr_median_n_spikes": _median_delta(candidate_table, swr_reference, "n_spikes"),
        "candidate_median_active_cell_count": _safe_median(candidate_table, "active_cell_count"),
        "swr_median_active_cell_count": _safe_median(swr_reference, "active_cell_count"),
        "candidate_minus_swr_median_active_cell_count": _median_delta(candidate_table, swr_reference, "active_cell_count"),
        "candidate_median_trajectory_posterior_entropy": _safe_median(candidate_table, "trajectory_posterior_entropy"),
        "swr_median_trajectory_posterior_entropy": _safe_median(swr_reference, "trajectory_posterior_entropy"),
        "candidate_minus_swr_median_trajectory_posterior_entropy": _median_delta(
            candidate_table,
            swr_reference,
            "trajectory_posterior_entropy",
        ),
        "candidate_median_decoded_path_length": _safe_median(candidate_table, "decoded_path_length"),
        "swr_median_decoded_path_length": _safe_median(swr_reference, "decoded_path_length"),
        "candidate_minus_swr_median_decoded_path_length": _median_delta(candidate_table, swr_reference, "decoded_path_length"),
        "candidate_median_decoded_speed": _safe_median(candidate_table, "decoded_speed"),
        "swr_median_decoded_speed": _safe_median(swr_reference, "decoded_speed"),
        "candidate_minus_swr_median_decoded_speed": _median_delta(candidate_table, swr_reference, "decoded_speed"),
        "candidate_median_duration_s": _safe_median(candidate_table, "duration_s"),
        "swr_median_duration_s": _safe_median(swr_reference, "duration_s"),
        "candidate_minus_swr_median_duration_s": _median_delta(candidate_table, swr_reference, "duration_s"),
        "candidate_median_distance_to_nearest_swr_s": _safe_median(candidate_table, "distance_to_nearest_swr_s"),
        "swr_median_distance_to_nearest_swr_s": _safe_median(swr_reference, "distance_to_nearest_swr_s"),
        "candidate_median_animal_speed_mean": _safe_median(candidate_table, "animal_speed_mean"),
        "swr_median_animal_speed_mean": _safe_median(swr_reference, "animal_speed_mean"),
        "candidate_minus_swr_median_animal_speed_mean": _median_delta(candidate_table, swr_reference, "animal_speed_mean"),
        "candidate_fraction_movement_spiking_like": _safe_fraction(movement_like, len(candidate_table)),
        "candidate_fraction_interesting": _safe_fraction(interesting, len(candidate_table)),
        "candidate_fraction_run": _safe_fraction(run_candidates, len(candidates)),
        "off_swr_best_trajectory_model_distribution": _format_model_distribution(model_distribution, "off_swr_candidate"),
        "swr_best_trajectory_model_distribution": _format_model_distribution(model_distribution, "swr_replay"),
        "off_swr_vs_swr_interpretation": "",
        "claim_should_narrow": False,
    }
    row["off_swr_vs_swr_interpretation"], row["claim_should_narrow"] = _interpret_off_swr_vs_swr(row)
    return pd.DataFrame([row], columns=list(CANDIDATE_VS_SWR_COLUMNS))


def _off_swr_run_state_window_table(
    decisions: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    required_models: Sequence[str],
    trajectory_models: Sequence[str],
    run_speed_threshold_cm_s: float = 5.0,
) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    off_swr = decisions[decisions["passes_known_swr_exclusion"].map(_as_bool)].copy()
    if off_swr.empty:
        return pd.DataFrame()

    lookup = _score_lookup(scores)
    swr_intervals = _real_swr_intervals(scores)
    required = tuple(str(model) for model in required_models)
    trajectory = tuple(str(model) for model in trajectory_models if str(model) in set(required))

    rows: list[dict[str, object]] = []
    for _, row in off_swr.iterrows():
        session = str(row["session"])
        event_index = int(row["event_index"])
        window_role = str(row["window_role"])
        null_index = int(row["null_index"])
        group = lookup.get((session, event_index, window_role, null_index), pd.DataFrame())
        best_trajectory = _best_model_row(group, row.get("best_trajectory_model"))

        start = _finite_or_nan(row.get("window_start_s"))
        end = _finite_or_nan(row.get("window_end_s"))
        duration = _finite_or_nan(row.get("window_duration_s"))
        distance_to_swr, overlaps_swr = _distance_to_nearest_interval(
            start,
            end,
            swr_intervals.get(session, np.empty((0, 2))),
        )
        animal_speed_mean = _first_numeric_from_columns(best_trajectory, TRIAGE_MEAN_SPEED_COLUMNS)
        animal_speed_median = _first_numeric_from_columns(best_trajectory, TRIAGE_MEDIAN_SPEED_COLUMNS)
        animal_speed_max = _first_numeric_from_columns(best_trajectory, TRIAGE_MAX_SPEED_COLUMNS)
        position_sample_count = _first_numeric_from_columns(best_trajectory, ("position_sample_count",))
        endpoint_x = _first_numeric_from_columns(best_trajectory, TRIAGE_ENDPOINT_X_COLUMNS)
        endpoint_y = _first_numeric_from_columns(best_trajectory, TRIAGE_ENDPOINT_Y_COLUMNS)
        start_x = _first_numeric_from_columns(best_trajectory, TRIAGE_START_X_COLUMNS)
        start_y = _first_numeric_from_columns(best_trajectory, TRIAGE_START_Y_COLUMNS)
        animal_x = _first_numeric_from_columns(best_trajectory, TRIAGE_ANIMAL_X_COLUMNS)
        animal_y = _first_numeric_from_columns(best_trajectory, TRIAGE_ANIMAL_Y_COLUMNS)
        decoded_start_to_end = _euclidean_distance((start_x, start_y), (endpoint_x, endpoint_y))
        decoded_endpoint_distance = _euclidean_distance((animal_x, animal_y), (endpoint_x, endpoint_y))
        decoded_path_length = _first_numeric_from_columns(best_trajectory, TRIAGE_PATH_LENGTH_COLUMNS)

        rows.append(
            {
                "comparison_scope": str(row.get("comparison_scope", "")),
                "window_set": "off_swr",
                "run_state": _run_or_immobility_state(
                    animal_speed_mean=animal_speed_mean,
                    animal_speed_median=animal_speed_median,
                    animal_speed_max=animal_speed_max,
                    run_speed_threshold_cm_s=run_speed_threshold_cm_s,
                ),
                "session": session,
                "rat": str(row.get("rat", _rat_from_session(session))),
                "event_index": event_index,
                "window_role": window_role,
                "null_index": null_index,
                "candidate_class": str(row.get("candidate_class", "")),
                "is_trajectory_family_candidate": bool(row.get("is_trajectory_family_candidate", False)),
                "trajectory_confident_claim": bool(row.get("trajectory_confident_claim", False)),
                "nontrajectory_confident_claim": bool(row.get("nontrajectory_confident_claim", False)),
                "window_start_s": start,
                "window_end_s": end,
                "duration_s": duration,
                "n_spikes": _finite_or_nan(row.get("n_spikes")),
                "active_cell_count": _finite_or_nan(row.get("active_cell_count")),
                "trajectory_family_margin": _finite_or_nan(row.get("trajectory_minus_nontrajectory_log_evidence")),
                "best_trajectory_model": str(row.get("best_trajectory_model", "")),
                "trajectory_confidence": _trajectory_confidence_from_scores(
                    group,
                    required_models=required,
                    trajectory_models=trajectory,
                ),
                "trajectory_posterior_entropy": _first_numeric_from_columns(best_trajectory, TRIAGE_ENTROPY_COLUMNS),
                "decoded_path_length": decoded_path_length,
                "decoded_speed": _decoded_speed(decoded_path_length, decoded_start_to_end, duration),
                "decoded_endpoint_distance": decoded_endpoint_distance,
                "decoded_start_to_end_distance": decoded_start_to_end,
                "animal_speed_mean": animal_speed_mean,
                "animal_speed_median": animal_speed_median,
                "animal_speed_max": animal_speed_max,
                "position_sample_count": position_sample_count,
                "distance_to_nearest_swr_s": distance_to_swr,
                "overlaps_known_swr": bool(overlaps_swr),
            }
        )
    return pd.DataFrame(rows)


def _summarize_run_state_stratum(
    frame: pd.DataFrame,
    *,
    comparison_scope: str,
    stratum: str,
    window_set: str,
    run_state: str,
) -> dict[str, object]:
    if frame.empty:
        return {
            "comparison_scope": comparison_scope,
            "stratum": stratum,
            "window_set": window_set,
            "run_state": run_state,
            "windows": 0,
            "trajectory_family_candidates": 0,
            "candidate_fraction": np.nan,
            "trajectory_confident_claims": 0,
            "nontrajectory_confident_claims": 0,
            "ambiguous_windows": 0,
            "incomplete_windows": 0,
            "mean_family_margin": np.nan,
            "median_family_margin": np.nan,
            "max_family_margin": np.nan,
            "median_trajectory_confidence": np.nan,
            "median_n_spikes": np.nan,
            "median_active_cell_count": np.nan,
            "median_trajectory_posterior_entropy": np.nan,
            "median_decoded_path_length": np.nan,
            "median_decoded_speed": np.nan,
            "median_duration_s": np.nan,
            "median_distance_to_nearest_swr_s": np.nan,
            "median_animal_speed_mean": np.nan,
            "best_trajectory_model_distribution": "",
        }

    candidates = int(frame["is_trajectory_family_candidate"].map(_as_bool).sum()) if "is_trajectory_family_candidate" in frame else 0
    margins = _numeric_series(frame, "trajectory_family_margin").dropna()
    return {
        "comparison_scope": comparison_scope,
        "stratum": stratum,
        "window_set": window_set,
        "run_state": run_state,
        "windows": int(len(frame)),
        "trajectory_family_candidates": candidates,
        "candidate_fraction": _safe_fraction(candidates, len(frame)),
        "trajectory_confident_claims": int(frame["trajectory_confident_claim"].map(_as_bool).sum())
        if "trajectory_confident_claim" in frame
        else candidates,
        "nontrajectory_confident_claims": int(frame["nontrajectory_confident_claim"].map(_as_bool).sum())
        if "nontrajectory_confident_claim" in frame
        else 0,
        "ambiguous_windows": int(frame["candidate_class"].astype(str).eq(AMBIGUOUS_CLASS).sum()) if "candidate_class" in frame else 0,
        "incomplete_windows": int(frame["candidate_class"].astype(str).eq(INCOMPLETE_CLASS).sum()) if "candidate_class" in frame else 0,
        "mean_family_margin": float(margins.mean()) if not margins.empty else np.nan,
        "median_family_margin": float(margins.median()) if not margins.empty else np.nan,
        "max_family_margin": float(margins.max()) if not margins.empty else np.nan,
        "median_trajectory_confidence": _safe_median(frame, "trajectory_confidence"),
        "median_n_spikes": _safe_median(frame, "n_spikes"),
        "median_active_cell_count": _safe_median(frame, "active_cell_count"),
        "median_trajectory_posterior_entropy": _safe_median(frame, "trajectory_posterior_entropy"),
        "median_decoded_path_length": _safe_median(frame, "decoded_path_length"),
        "median_decoded_speed": _safe_median(frame, "decoded_speed"),
        "median_duration_s": _safe_median(frame, "duration_s"),
        "median_distance_to_nearest_swr_s": _safe_median(frame, "distance_to_nearest_swr_s"),
        "median_animal_speed_mean": _safe_median(frame, "animal_speed_mean"),
        "best_trajectory_model_distribution": _format_model_distribution_from_frame(frame),
    }


def off_swr_run_state_stratified_summary(
    decisions: pd.DataFrame,
    scores: pd.DataFrame,
    candidate_vs_swr_window_table: pd.DataFrame,
    *,
    required_models: Sequence[str],
    trajectory_models: Sequence[str],
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    run_speed_threshold_cm_s: float = 5.0,
    off_swr_windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if off_swr_windows is None:
        off_swr_windows = _off_swr_run_state_window_table(
            decisions,
            scores,
            required_models=required_models,
            trajectory_models=trajectory_models,
            run_speed_threshold_cm_s=run_speed_threshold_cm_s,
        )
    comparison_scope = str(_first_value(decisions, "comparison_scope"))
    if (comparison_scope == "nan" or not comparison_scope) and not candidate_vs_swr_window_table.empty:
        comparison_scope = str(_first_value(candidate_vs_swr_window_table, "comparison_scope"))

    rows: list[dict[str, object]] = []
    for run_state, stratum in (
        ("immobile", "off_swr_immobile_windows"),
        ("run", "off_swr_running_windows"),
        ("unknown_speed", "off_swr_unknown_speed_windows"),
    ):
        subset = off_swr_windows[off_swr_windows["run_state"].astype(str).eq(run_state)].copy() if not off_swr_windows.empty else pd.DataFrame()
        rows.append(
            _summarize_run_state_stratum(
                subset,
                comparison_scope=comparison_scope,
                stratum=stratum,
                window_set="off_swr",
                run_state=run_state,
            )
        )

    swr = (
        candidate_vs_swr_window_table[candidate_vs_swr_window_table["window_set"].astype(str).eq("swr_replay")].copy()
        if not candidate_vs_swr_window_table.empty
        else pd.DataFrame()
    )
    if not swr.empty:
        margins = _numeric_series(swr, "trajectory_family_margin")
        swr["is_trajectory_family_candidate"] = margins >= float(margin_threshold)
        swr["trajectory_confident_claim"] = margins >= float(margin_threshold)
        swr["nontrajectory_confident_claim"] = margins <= -float(margin_threshold)
        swr["candidate_class"] = "swr_replay_reference"
    rows.append(
        _summarize_run_state_stratum(
            swr,
            comparison_scope=comparison_scope,
            stratum="swr_replay_windows",
            window_set="swr_replay",
            run_state="swr_replay",
        )
    )
    return pd.DataFrame(rows, columns=list(RUN_STATE_STRATIFIED_SUMMARY_COLUMNS))


def off_swr_run_state_specificity_summary(stratified: pd.DataFrame) -> pd.DataFrame:
    if stratified.empty:
        return _empty_frame(RUN_STATE_SPECIFICITY_COLUMNS)

    def int_value(stratum: str, column: str) -> int:
        subset = stratified[stratified["stratum"].astype(str).eq(stratum)]
        if subset.empty:
            return 0
        value = _finite_or_nan(subset.iloc[0].get(column))
        return int(value) if np.isfinite(value) else 0

    comparison_scope = str(_first_value(stratified, "comparison_scope"))
    immobile_windows = int_value("off_swr_immobile_windows", "windows")
    running_windows = int_value("off_swr_running_windows", "windows")
    unknown_windows = int_value("off_swr_unknown_speed_windows", "windows")
    immobile_candidates = int_value("off_swr_immobile_windows", "trajectory_family_candidates")
    running_candidates = int_value("off_swr_running_windows", "trajectory_family_candidates")
    unknown_candidates = int_value("off_swr_unknown_speed_windows", "trajectory_family_candidates")
    swr_reference = int_value("swr_replay_windows", "windows")
    off_swr_windows = immobile_windows + running_windows + unknown_windows
    off_swr_candidates = immobile_candidates + running_candidates + unknown_candidates
    speed_evaluable_windows = immobile_windows + running_windows

    if off_swr_candidates == 0:
        interpretation = "no_off_swr_trajectory_candidates"
        claim_should_narrow = False
    elif speed_evaluable_windows == 0:
        interpretation = "speed_unavailable_for_off_swr_specificity"
        claim_should_narrow = False
    elif immobile_candidates > 0:
        interpretation = "immobile_off_swr_candidates_present"
        claim_should_narrow = False
    elif running_candidates > 0:
        interpretation = "candidate_signal_concentrated_in_running_windows"
        claim_should_narrow = True
    else:
        interpretation = "candidate_signal_speed_unknown"
        claim_should_narrow = False

    row = {
        "comparison_scope": comparison_scope,
        "off_swr_windows": off_swr_windows,
        "off_swr_immobile_windows": immobile_windows,
        "off_swr_running_windows": running_windows,
        "off_swr_unknown_speed_windows": unknown_windows,
        "off_swr_candidates": off_swr_candidates,
        "immobile_off_swr_candidates": immobile_candidates,
        "running_off_swr_candidates": running_candidates,
        "unknown_speed_off_swr_candidates": unknown_candidates,
        "immobile_candidate_fraction": _safe_fraction(immobile_candidates, immobile_windows),
        "running_candidate_fraction": _safe_fraction(running_candidates, running_windows),
        "unknown_speed_candidate_fraction": _safe_fraction(unknown_candidates, unknown_windows),
        "swr_reference_windows": swr_reference,
        "immobile_candidate_signal_present": bool(immobile_candidates > 0),
        "run_state_specificity_interpretation": interpretation,
        "claim_should_narrow_for_run_state": bool(claim_should_narrow),
    }
    return pd.DataFrame([row], columns=list(RUN_STATE_SPECIFICITY_COLUMNS))


def _nearest_swr_exclusion_label(radius_s: float) -> str:
    milliseconds = int(round(float(radius_s) * 1000.0))
    return f"exclude_within_{milliseconds}ms"


def _nearest_swr_exclusion_interpretation(
    *,
    windows_after: int,
    candidates_after: int,
    fraction_retained: float,
    radius_s: float,
) -> tuple[str, bool]:
    if windows_after == 0:
        return "no_evaluable_windows_after_exclusion", False
    if candidates_after == 0:
        return "candidate_signal_vanishes_after_nearest_swr_exclusion", True
    if float(radius_s) >= 0.5 and np.isfinite(fraction_retained) and fraction_retained < 0.25:
        return "candidate_signal_substantially_attenuates_after_nearest_swr_exclusion", True
    return "candidate_signal_persists_after_nearest_swr_exclusion", False


def off_swr_nearest_swr_exclusion_summary(
    decisions: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    required_models: Sequence[str],
    trajectory_models: Sequence[str],
    exclusion_radii_s: Sequence[float] = DEFAULT_NEAREST_SWR_EXCLUSION_RADII_S,
    run_speed_threshold_cm_s: float = 5.0,
    off_swr_windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if off_swr_windows is None:
        off_swr_windows = _off_swr_run_state_window_table(
            decisions,
            scores,
            required_models=required_models,
            trajectory_models=trajectory_models,
            run_speed_threshold_cm_s=run_speed_threshold_cm_s,
        )
    if off_swr_windows.empty:
        return _empty_frame(NEAREST_SWR_EXCLUSION_COLUMNS)

    comparison_scope = str(_first_value(off_swr_windows, "comparison_scope"))
    distances = _numeric_series(off_swr_windows, "distance_to_nearest_swr_s")
    candidate_mask = off_swr_windows["is_trajectory_family_candidate"].map(_as_bool)
    before_windows = int(len(off_swr_windows))
    before_candidates = int(candidate_mask.sum())
    evaluable = distances.notna()
    evaluable_windows = int(evaluable.sum())

    rows: list[dict[str, object]] = []
    for radius_s in exclusion_radii_s:
        keep = evaluable & (distances >= float(radius_s))
        after = off_swr_windows[keep].copy()
        after_candidate_mask = after["is_trajectory_family_candidate"].map(_as_bool) if not after.empty else pd.Series(dtype=bool)
        after_candidates = int(after_candidate_mask.sum()) if not after.empty else 0
        excluded = off_swr_windows[evaluable & ~keep].copy()
        excluded_candidate_mask = excluded["is_trajectory_family_candidate"].map(_as_bool) if not excluded.empty else pd.Series(dtype=bool)
        candidate_fraction_after = _safe_fraction(after_candidates, len(after))
        fraction_candidates_retained = _safe_fraction(after_candidates, before_candidates)
        candidate_rows = after[after_candidate_mask].copy() if not after.empty else pd.DataFrame()
        interpretation, should_narrow = _nearest_swr_exclusion_interpretation(
            windows_after=int(len(after)),
            candidates_after=after_candidates,
            fraction_retained=fraction_candidates_retained,
            radius_s=float(radius_s),
        )
        rows.append(
            {
                "comparison_scope": comparison_scope,
                "exclusion_radius_s": float(radius_s),
                "exclusion_label": _nearest_swr_exclusion_label(float(radius_s)),
                "off_swr_windows_before_exclusion": before_windows,
                "candidate_windows_before_exclusion": before_candidates,
                "candidate_fraction_before_exclusion": _safe_fraction(before_candidates, before_windows),
                "evaluable_distance_windows": evaluable_windows,
                "windows_after_exclusion": int(len(after)),
                "candidate_windows_after_exclusion": after_candidates,
                "candidate_fraction_after_exclusion": candidate_fraction_after,
                "windows_excluded": int(len(excluded)),
                "candidate_windows_excluded": int(excluded_candidate_mask.sum()) if not excluded.empty else 0,
                "fraction_windows_retained": _safe_fraction(len(after), evaluable_windows),
                "fraction_candidates_retained": fraction_candidates_retained,
                "candidate_sessions_after_exclusion": int(candidate_rows["session"].nunique()) if not candidate_rows.empty else 0,
                "candidate_rats_after_exclusion": int(candidate_rows["rat"].nunique()) if not candidate_rows.empty else 0,
                "median_distance_to_nearest_swr_s_after_exclusion": _safe_median(after, "distance_to_nearest_swr_s"),
                "median_family_margin_after_exclusion": _safe_median(after, "trajectory_family_margin"),
                "mean_family_margin_after_exclusion": float(_numeric_series(after, "trajectory_family_margin").mean())
                if not after.empty and _numeric_series(after, "trajectory_family_margin").notna().any()
                else np.nan,
                "median_candidate_family_margin_after_exclusion": _safe_median(candidate_rows, "trajectory_family_margin"),
                "nearest_swr_exclusion_interpretation": interpretation
                if evaluable_windows > 0
                else "distance_unavailable_for_nearest_swr_specificity",
                "claim_should_narrow_for_nearest_swr": bool(should_narrow) if evaluable_windows > 0 else False,
            }
        )
    return pd.DataFrame(rows, columns=list(NEAREST_SWR_EXCLUSION_COLUMNS))


def off_swr_nearest_swr_specificity_summary(exclusion_summary: pd.DataFrame) -> pd.DataFrame:
    if exclusion_summary.empty:
        return _empty_frame(NEAREST_SWR_SPECIFICITY_COLUMNS)

    def row_for_radius(radius_s: float) -> pd.Series | None:
        distances = _numeric_series(exclusion_summary, "exclusion_radius_s")
        subset = exclusion_summary[np.isclose(distances.to_numpy(dtype=float), float(radius_s), equal_nan=False)]
        if subset.empty:
            return None
        return subset.iloc[0]

    radius_500 = row_for_radius(0.5)
    radius_1s = row_for_radius(1.0)
    first = exclusion_summary.iloc[0]
    evaluable = int(_finite_or_nan(first.get("evaluable_distance_windows"))) if np.isfinite(_finite_or_nan(first.get("evaluable_distance_windows"))) else 0
    candidate_after_500 = int(_finite_or_nan(radius_500.get("candidate_windows_after_exclusion"))) if radius_500 is not None else 0
    candidate_after_1s = int(_finite_or_nan(radius_1s.get("candidate_windows_after_exclusion"))) if radius_1s is not None else 0
    retention_500 = _finite_or_nan(radius_500.get("fraction_candidates_retained")) if radius_500 is not None else np.nan
    retention_1s = _finite_or_nan(radius_1s.get("fraction_candidates_retained")) if radius_1s is not None else np.nan
    should_narrow_500 = bool(_as_bool(radius_500.get("claim_should_narrow_for_nearest_swr"))) if radius_500 is not None else False
    should_narrow_1s = bool(_as_bool(radius_1s.get("claim_should_narrow_for_nearest_swr"))) if radius_1s is not None else False

    if evaluable == 0:
        interpretation = "distance_unavailable_for_nearest_swr_specificity"
        should_narrow = False
    elif candidate_after_500 > 0 and candidate_after_1s > 0:
        interpretation = "candidate_signal_persists_beyond_500ms_and_1s"
        should_narrow = False
    elif candidate_after_500 > 0:
        interpretation = "candidate_signal_persists_beyond_500ms_but_not_1s"
        should_narrow = bool(should_narrow_1s)
    else:
        interpretation = "candidate_signal_vanishes_by_500ms_nearest_swr_exclusion"
        should_narrow = True

    row = {
        "comparison_scope": str(first.get("comparison_scope", "")),
        "off_swr_windows": int(_finite_or_nan(first.get("off_swr_windows_before_exclusion"))),
        "candidate_windows": int(_finite_or_nan(first.get("candidate_windows_before_exclusion"))),
        "candidate_fraction": _finite_or_nan(first.get("candidate_fraction_before_exclusion")),
        "evaluable_distance_windows": evaluable,
        "candidate_windows_after_500ms_exclusion": candidate_after_500,
        "candidate_fraction_after_500ms_exclusion": _finite_or_nan(radius_500.get("candidate_fraction_after_exclusion"))
        if radius_500 is not None
        else np.nan,
        "candidate_retention_after_500ms_exclusion": retention_500,
        "candidate_windows_after_1s_exclusion": candidate_after_1s,
        "candidate_fraction_after_1s_exclusion": _finite_or_nan(radius_1s.get("candidate_fraction_after_exclusion"))
        if radius_1s is not None
        else np.nan,
        "candidate_retention_after_1s_exclusion": retention_1s,
        "nearest_swr_specificity_interpretation": interpretation,
        "claim_should_narrow_for_nearest_swr": bool(should_narrow or should_narrow_500),
    }
    return pd.DataFrame([row], columns=list(NEAREST_SWR_SPECIFICITY_COLUMNS))


def _tier_summary_row(frame: pd.DataFrame, *, tier: str, threshold: float, comparison_scope: str) -> dict[str, object]:
    candidate_mask = _tier_candidate_mask(frame, threshold)
    candidates = frame[candidate_mask].copy() if not frame.empty else pd.DataFrame()
    immobile = frame[frame["run_state"].astype(str).eq("immobile")].copy() if not frame.empty else pd.DataFrame()
    running = frame[frame["run_state"].astype(str).eq("run")].copy() if not frame.empty else pd.DataFrame()
    unknown = frame[frame["run_state"].astype(str).eq("unknown_speed")].copy() if not frame.empty else pd.DataFrame()
    distance = _numeric_series(frame, "distance_to_nearest_swr_s") if not frame.empty else pd.Series(dtype=float)
    after_500 = frame[distance.notna() & (distance >= 0.5)].copy() if not frame.empty else pd.DataFrame()
    after_1s = frame[distance.notna() & (distance >= 1.0)].copy() if not frame.empty else pd.DataFrame()

    def candidate_count(subset: pd.DataFrame) -> int:
        return int(_tier_candidate_mask(subset, threshold).sum()) if not subset.empty else 0

    return {
        "comparison_scope": comparison_scope,
        "candidate_tier": tier,
        "tier_margin_threshold": float(threshold),
        "off_swr_windows": int(len(frame)),
        "candidate_windows": int(candidate_mask.sum()) if not frame.empty else 0,
        "candidate_fraction": _safe_fraction(int(candidate_mask.sum()) if not frame.empty else 0, len(frame)),
        "candidate_sessions": int(candidates["session"].nunique()) if not candidates.empty else 0,
        "candidate_rats": int(candidates["rat"].nunique()) if not candidates.empty else 0,
        "immobile_windows": int(len(immobile)),
        "immobile_candidate_windows": candidate_count(immobile),
        "immobile_candidate_fraction": _safe_fraction(candidate_count(immobile), len(immobile)),
        "running_windows": int(len(running)),
        "running_candidate_windows": candidate_count(running),
        "running_candidate_fraction": _safe_fraction(candidate_count(running), len(running)),
        "unknown_speed_windows": int(len(unknown)),
        "unknown_speed_candidate_windows": candidate_count(unknown),
        "unknown_speed_candidate_fraction": _safe_fraction(candidate_count(unknown), len(unknown)),
        "candidate_windows_after_500ms_swr_exclusion": candidate_count(after_500),
        "candidate_fraction_after_500ms_swr_exclusion": _safe_fraction(candidate_count(after_500), len(after_500)),
        "candidate_windows_after_1s_swr_exclusion": candidate_count(after_1s),
        "candidate_fraction_after_1s_swr_exclusion": _safe_fraction(candidate_count(after_1s), len(after_1s)),
        "median_candidate_family_margin": _safe_median(candidates, "trajectory_family_margin"),
        "best_trajectory_model_distribution": _format_model_distribution_from_frame(candidates),
    }


def off_swr_candidate_tier_threshold_summary(
    off_swr_windows: pd.DataFrame,
    *,
    thresholds: Sequence[tuple[str, float]] = DEFAULT_CANDIDATE_TIER_THRESHOLDS,
) -> pd.DataFrame:
    if off_swr_windows.empty:
        return _empty_frame(CANDIDATE_TIER_THRESHOLD_SUMMARY_COLUMNS)
    comparison_scope = str(_first_value(off_swr_windows, "comparison_scope"))
    rows = [
        _tier_summary_row(off_swr_windows, tier=str(tier), threshold=float(threshold), comparison_scope=comparison_scope)
        for tier, threshold in thresholds
    ]
    return pd.DataFrame(rows, columns=list(CANDIDATE_TIER_THRESHOLD_SUMMARY_COLUMNS))


def off_swr_candidate_tier_group_summary(
    off_swr_windows: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    thresholds: Sequence[tuple[str, float]] = DEFAULT_CANDIDATE_TIER_THRESHOLDS,
) -> pd.DataFrame:
    if off_swr_windows.empty:
        return _empty_frame(CANDIDATE_TIER_GROUP_SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    for key, group in off_swr_windows.groupby(list(group_cols), sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        comparison_scope = str(_first_value(group, "comparison_scope"))
        for tier, threshold in thresholds:
            candidate_mask = _tier_candidate_mask(group, float(threshold))
            candidates = group[candidate_mask].copy()
            immobile = group[group["run_state"].astype(str).eq("immobile")]
            running = group[group["run_state"].astype(str).eq("run")]
            unknown = group[group["run_state"].astype(str).eq("unknown_speed")]
            distance = _numeric_series(group, "distance_to_nearest_swr_s")
            after_500 = group[distance.notna() & (distance >= 0.5)]
            after_1s = group[distance.notna() & (distance >= 1.0)]
            row = {
                "comparison_scope": comparison_scope,
                "rat": str(_first_value(group, "rat")),
                "session": str(_first_value(group, "session")) if "session" in group_cols else "all",
                "candidate_tier": str(tier),
                "tier_margin_threshold": float(threshold),
                "off_swr_windows": int(len(group)),
                "candidate_windows": int(candidate_mask.sum()),
                "candidate_fraction": _safe_fraction(int(candidate_mask.sum()), len(group)),
                "immobile_candidate_windows": int(_tier_candidate_mask(immobile, float(threshold)).sum()) if not immobile.empty else 0,
                "running_candidate_windows": int(_tier_candidate_mask(running, float(threshold)).sum()) if not running.empty else 0,
                "unknown_speed_candidate_windows": int(_tier_candidate_mask(unknown, float(threshold)).sum()) if not unknown.empty else 0,
                "candidate_windows_after_500ms_swr_exclusion": int(_tier_candidate_mask(after_500, float(threshold)).sum())
                if not after_500.empty
                else 0,
                "candidate_windows_after_1s_swr_exclusion": int(_tier_candidate_mask(after_1s, float(threshold)).sum())
                if not after_1s.empty
                else 0,
                "median_candidate_family_margin": _safe_median(candidates, "trajectory_family_margin"),
            }
            row.update(base)
            rows.append(row)
    columns = [column for column in CANDIDATE_TIER_GROUP_SUMMARY_COLUMNS if column in {"comparison_scope", *group_cols} or column not in group_cols]
    return pd.DataFrame(rows)[columns]


def off_swr_candidate_tier_nearest_swr_exclusion_summary(
    off_swr_windows: pd.DataFrame,
    *,
    thresholds: Sequence[tuple[str, float]] = DEFAULT_CANDIDATE_TIER_THRESHOLDS,
    exclusion_radii_s: Sequence[float] = DEFAULT_NEAREST_SWR_EXCLUSION_RADII_S,
) -> pd.DataFrame:
    if off_swr_windows.empty:
        return _empty_frame(CANDIDATE_TIER_NEAREST_SWR_EXCLUSION_COLUMNS)
    comparison_scope = str(_first_value(off_swr_windows, "comparison_scope"))
    distance = _numeric_series(off_swr_windows, "distance_to_nearest_swr_s")
    rows: list[dict[str, object]] = []
    for tier, threshold in thresholds:
        before_candidates = int(_tier_candidate_mask(off_swr_windows, float(threshold)).sum())
        for radius_s in exclusion_radii_s:
            after = off_swr_windows[distance.notna() & (distance >= float(radius_s))].copy()
            candidate_mask = _tier_candidate_mask(after, float(threshold)) if not after.empty else pd.Series(dtype=bool)
            candidates = after[candidate_mask].copy() if not after.empty else pd.DataFrame()
            candidate_count = int(candidate_mask.sum()) if not after.empty else 0
            rows.append(
                {
                    "comparison_scope": comparison_scope,
                    "candidate_tier": str(tier),
                    "tier_margin_threshold": float(threshold),
                    "exclusion_radius_s": float(radius_s),
                    "exclusion_label": _nearest_swr_exclusion_label(float(radius_s)),
                    "windows_after_exclusion": int(len(after)),
                    "candidate_windows_after_exclusion": candidate_count,
                    "candidate_fraction_after_exclusion": _safe_fraction(candidate_count, len(after)),
                    "candidate_retention_after_exclusion": _safe_fraction(candidate_count, before_candidates),
                    "median_candidate_family_margin_after_exclusion": _safe_median(candidates, "trajectory_family_margin"),
                }
            )
    return pd.DataFrame(rows, columns=list(CANDIDATE_TIER_NEAREST_SWR_EXCLUSION_COLUMNS))


def off_swr_high_specificity_candidate_table(
    candidate_table: pd.DataFrame,
    *,
    strong_threshold: float = 50.0,
    extreme_threshold: float = 100.0,
    swr_exclusion_radius_s: float = 1.0,
) -> pd.DataFrame:
    if candidate_table.empty:
        return _empty_frame(HIGH_SPECIFICITY_CANDIDATE_COLUMNS)
    table = candidate_table.copy()
    margins = _numeric_series(table, "trajectory_family_margin")
    distances = _numeric_series(table, "distance_to_nearest_swr_s")
    speed_values = _numeric_series(table, "animal_speed_mean")
    run_state = table["run_or_immobility_state"].astype(str) if "run_or_immobility_state" in table else pd.Series("", index=table.index)
    specificity_label = (
        table["candidate_specificity_label"].astype(str)
        if "candidate_specificity_label" in table
        else pd.Series("", index=table.index)
    )
    table["passes_strong_tier"] = margins >= float(strong_threshold)
    table["passes_extreme_tier"] = margins >= float(extreme_threshold)
    table["passes_500ms_swr_exclusion"] = distances.notna() & (distances >= 0.5)
    table["passes_1s_swr_exclusion"] = distances.notna() & (distances >= float(swr_exclusion_radius_s))
    table["speed_available"] = speed_values.notna()
    table["passes_immobility_filter"] = run_state.eq("immobile")
    table["passes_specificity_label_filter"] = specificity_label.eq(INTERESTING_CANDIDATE_LABEL)
    speed_evaluable = bool(table["speed_available"].any())
    table["passes_high_specificity_promotion_filter"] = (
        table["passes_strong_tier"]
        & table["passes_1s_swr_exclusion"]
        & table["passes_specificity_label_filter"]
        & (table["passes_immobility_filter"] if speed_evaluable else False)
    )
    if speed_evaluable:
        table["promotion_limitation"] = np.where(
            table["passes_high_specificity_promotion_filter"],
            "",
            "fails_strong_or_distance_or_immobility_or_specificity_filter",
        )
    else:
        table["promotion_limitation"] = "speed_unavailable_for_immobility_filter"
    tier_distance = table["passes_strong_tier"] & table["passes_1s_swr_exclusion"]
    table["high_specificity_label"] = np.select(
        [
            table["passes_high_specificity_promotion_filter"],
            tier_distance & ~table["passes_specificity_label_filter"],
            tier_distance,
        ],
        [
            "promotion_ready_high_specificity_candidate",
            "tier_distance_candidate_movement_spiking_or_low_information",
            "tier_distance_candidate_speed_unavailable_or_nonimmobile",
        ],
        default="below_high_specificity_filter",
    )
    table = table[tier_distance].copy()
    if table.empty:
        return _empty_frame(HIGH_SPECIFICITY_CANDIDATE_COLUMNS)
    return table[[column for column in HIGH_SPECIFICITY_CANDIDATE_COLUMNS if column in table.columns]]


def off_swr_promotion_readiness_summary(
    candidate_table: pd.DataFrame,
    high_specificity_candidates: pd.DataFrame,
    nearest_swr_specificity: pd.DataFrame,
    run_state_specificity: pd.DataFrame,
    *,
    strong_threshold: float = 50.0,
    extreme_threshold: float = 100.0,
) -> pd.DataFrame:
    comparison_scope = str(_first_value(candidate_table, "comparison_scope")) if not candidate_table.empty else ""
    margins = _numeric_series(candidate_table, "trajectory_family_margin")
    distances = _numeric_series(candidate_table, "distance_to_nearest_swr_s")
    run_state = candidate_table["run_or_immobility_state"].astype(str) if "run_or_immobility_state" in candidate_table else pd.Series(dtype=str)
    speed_available = _numeric_series(candidate_table, "animal_speed_mean").notna() if not candidate_table.empty else pd.Series(dtype=bool)
    strong_mask = margins >= float(strong_threshold)
    extreme_mask = margins >= float(extreme_threshold)
    strong_after_500 = strong_mask & distances.notna() & (distances >= 0.5)
    strong_after_1s = strong_mask & distances.notna() & (distances >= 1.0)
    speed_evaluable_count = int(speed_available.sum()) if not speed_available.empty else 0
    strong_immobile = strong_mask & run_state.eq("immobile") if not candidate_table.empty else pd.Series(dtype=bool)
    high_specificity_ready = (
        high_specificity_candidates["passes_high_specificity_promotion_filter"].map(_as_bool)
        if not high_specificity_candidates.empty and "passes_high_specificity_promotion_filter" in high_specificity_candidates
        else pd.Series(dtype=bool)
    )
    nearest_interpretation = (
        str(_first_value(nearest_swr_specificity, "nearest_swr_specificity_interpretation")) if not nearest_swr_specificity.empty else ""
    )
    run_state_interpretation = (
        str(_first_value(run_state_specificity, "run_state_specificity_interpretation")) if not run_state_specificity.empty else ""
    )
    nearest_narrow = (
        bool(nearest_swr_specificity["claim_should_narrow_for_nearest_swr"].map(_as_bool).any())
        if not nearest_swr_specificity.empty and "claim_should_narrow_for_nearest_swr" in nearest_swr_specificity
        else False
    )
    run_state_narrow = (
        bool(run_state_specificity["claim_should_narrow_for_run_state"].map(_as_bool).any())
        if not run_state_specificity.empty and "claim_should_narrow_for_run_state" in run_state_specificity
        else False
    )

    strong_count = int(strong_mask.sum()) if not strong_mask.empty else 0
    strong_after_1s_count = int(strong_after_1s.sum()) if not strong_after_1s.empty else 0
    high_specificity_count = int(high_specificity_ready.sum()) if not high_specificity_ready.empty else 0
    if strong_count == 0:
        status = "exploratory_no_strong_candidates"
        guidance = "Do not claim off-SWR replay; no strong-tier off-SWR candidates were detected."
    elif nearest_narrow or strong_after_1s_count == 0:
        status = "exploratory_nearest_swr_contamination_risk"
        guidance = "Keep the result exploratory; strong candidates do not survive the 1 s nearest-SWR exclusion cleanly."
    elif speed_evaluable_count == 0:
        status = "exploratory_speed_unavailable"
        guidance = "Do not make an immobility-specific off-SWR replay claim until animal-speed/run-state fields are available."
    elif run_state_narrow or int(strong_immobile.sum()) == 0:
        status = "exploratory_movement_decoding_risk"
        guidance = "Narrow the claim; strong candidates are not established in immobile non-SWR windows."
    elif high_specificity_count > 0:
        status = "ready_for_off_swr_replay_candidate_claim"
        guidance = "Paper-safe wording can describe high-specificity off-SWR trajectory candidates, pending independent behavior/LFP validation."
    else:
        status = "exploratory_high_specificity_filter_failed"
        guidance = "Keep the result exploratory; the combined tier, distance, and immobility filters did not yield promotable candidates."

    row = {
        "comparison_scope": comparison_scope,
        "promotion_status": status,
        "promotion_ready": bool(status == "ready_for_off_swr_replay_candidate_claim"),
        "off_swr_candidate_windows": int(len(candidate_table)),
        "strong_candidate_windows": strong_count,
        "extreme_candidate_windows": int(extreme_mask.sum()) if not extreme_mask.empty else 0,
        "strong_candidates_after_500ms_swr_exclusion": int(strong_after_500.sum()) if not strong_after_500.empty else 0,
        "strong_candidates_after_1s_swr_exclusion": strong_after_1s_count,
        "speed_evaluable_candidate_windows": speed_evaluable_count,
        "strong_immobile_candidate_windows": int(strong_immobile.sum()) if not strong_immobile.empty else 0,
        "high_specificity_candidate_windows": high_specificity_count,
        "nearest_swr_specificity_interpretation": nearest_interpretation,
        "run_state_specificity_interpretation": run_state_interpretation,
        "paper_claim_guidance": guidance,
    }
    return pd.DataFrame([row], columns=list(PROMOTION_READINESS_COLUMNS))


def off_swr_speed_coverage_summary(
    off_swr_windows: pd.DataFrame,
    candidate_table: pd.DataFrame,
    candidate_vs_swr_window_table: pd.DataFrame,
    promotion_readiness: pd.DataFrame,
    *,
    strong_threshold: float = 50.0,
) -> pd.DataFrame:
    comparison_scope = str(_first_value(off_swr_windows, "comparison_scope")) if not off_swr_windows.empty else ""

    def speed_count(frame: pd.DataFrame) -> int:
        return int(_numeric_series(frame, "animal_speed_mean").notna().sum()) if not frame.empty else 0

    def position_count(frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        if "position_sample_count" not in frame.columns:
            return 0
        return int((_numeric_series(frame, "position_sample_count") > 0).sum())

    swr = (
        candidate_vs_swr_window_table[candidate_vs_swr_window_table["window_set"].astype(str).eq("swr_replay")].copy()
        if not candidate_vs_swr_window_table.empty and "window_set" in candidate_vs_swr_window_table
        else pd.DataFrame()
    )
    candidate_speed = speed_count(candidate_table)
    strong_candidates = candidate_table[_numeric_series(candidate_table, "trajectory_family_margin") >= float(strong_threshold)].copy()
    strong_speed = speed_count(strong_candidates)
    off_swr_speed = speed_count(off_swr_windows)
    off_swr_count = int(len(off_swr_windows))
    candidate_count = int(len(candidate_table))
    strong_count = int(len(strong_candidates))
    promotion_status = str(_first_value(promotion_readiness, "promotion_status")) if not promotion_readiness.empty else ""
    if off_swr_count == 0:
        status = "no_off_swr_windows"
        ready = False
    elif off_swr_speed == 0:
        status = "speed_unavailable"
        ready = False
    elif candidate_count > 0 and candidate_speed == 0:
        status = "candidate_speed_unavailable"
        ready = False
    elif strong_count > 0 and strong_speed == 0:
        status = "strong_candidate_speed_unavailable"
        ready = False
    else:
        status = "speed_available_for_promotion_gate"
        ready = True
    row = {
        "comparison_scope": comparison_scope,
        "off_swr_windows": off_swr_count,
        "off_swr_windows_with_position_samples": position_count(off_swr_windows),
        "off_swr_windows_with_speed": off_swr_speed,
        "off_swr_speed_coverage_fraction": _safe_fraction(off_swr_speed, off_swr_count),
        "swr_reference_windows": int(len(swr)),
        "swr_reference_windows_with_speed": speed_count(swr),
        "candidate_windows": candidate_count,
        "candidate_windows_with_speed": candidate_speed,
        "candidate_speed_coverage_fraction": _safe_fraction(candidate_speed, candidate_count),
        "strong_candidate_windows": strong_count,
        "strong_candidate_windows_with_speed": strong_speed,
        "strong_candidate_speed_coverage_fraction": _safe_fraction(strong_speed, strong_count),
        "promotion_status": promotion_status,
        "speed_coverage_status": status,
        "speed_coverage_ready": bool(ready),
    }
    return pd.DataFrame([row], columns=list(SPEED_COVERAGE_COLUMNS))


def off_swr_behavior_lfp_summary(
    decisions: pd.DataFrame,
    *,
    behavior_lfp_columns: Sequence[str] = DEFAULT_BEHAVIOR_LFP_COLUMNS,
) -> pd.DataFrame:
    if decisions.empty:
        return _empty_frame(BEHAVIOR_LFP_COLUMNS)

    rows: list[dict[str, object]] = []
    for feature in behavior_lfp_columns:
        if feature not in decisions.columns:
            rows.append(
                {
                    "feature": feature,
                    "feature_available": False,
                    "candidate_class": "all",
                    "windows": int(len(decisions)),
                    "non_null_windows": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )
            continue
        working = decisions[["candidate_class", feature]].copy()
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
        if working[feature].notna().sum() == 0:
            rows.append(
                {
                    "feature": feature,
                    "feature_available": False,
                    "candidate_class": "all",
                    "windows": int(len(decisions)),
                    "non_null_windows": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )
            continue
        for candidate_class, group in working.groupby("candidate_class", sort=True):
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            rows.append(
                {
                    "feature": feature,
                    "feature_available": True,
                    "candidate_class": str(candidate_class),
                    "windows": int(len(group)),
                    "non_null_windows": int(len(values)),
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "min": float(values.min()) if not values.empty else np.nan,
                    "max": float(values.max()) if not values.empty else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=list(BEHAVIOR_LFP_COLUMNS))


def off_swr_discovery_gate_summary(
    decisions: pd.DataFrame,
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    behavior_lfp: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str, required_for_overall: bool = True) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "required_for_overall": bool(required_for_overall),
            }
        )

    allowed_classes = {
        TRAJECTORY_CANDIDATE_CLASS,
        STATIC_NONTRAJECTORY_CLASS,
        AMBIGUOUS_CLASS,
        INCOMPLETE_CLASS,
        EXCLUDED_SWR_OVERLAP_CLASS,
    }
    off_swr_windows = int(decisions["passes_known_swr_exclusion"].map(_as_bool).sum()) if not decisions.empty else 0
    complete_windows = int(decisions["required_models_complete"].map(_as_bool).sum()) if not decisions.empty else 0
    excluded_candidates = (
        candidates[~candidates["passes_known_swr_exclusion"].map(_as_bool)]
        if not candidates.empty and "passes_known_swr_exclusion" in candidates.columns
        else pd.DataFrame()
    )

    add("off_swr_windows_present", off_swr_windows > 0, off_swr_windows, "at least one matched off-SWR window was scored")
    add(
        "required_model_gate_applied",
        complete_windows > 0,
        f"{complete_windows}/{len(decisions)}",
        "at least one off-SWR window has complete required model evidence",
    )
    add(
        "candidate_classification_reported",
        (not decisions.empty) and set(decisions["candidate_class"].astype(str)).issubset(allowed_classes),
        " ".join(sorted(set(decisions["candidate_class"].astype(str)))) if not decisions.empty else "",
        "all off-SWR windows are assigned a discovery class",
    )
    add(
        "known_swr_overlap_removed_from_candidates",
        excluded_candidates.empty,
        int(len(excluded_candidates)),
        "trajectory-family candidate table contains only windows marked off_swr",
    )
    add(
        "clusters_assigned_for_candidates",
        candidates.empty or (not clusters.empty and int(clusters["window_count"].sum()) == len(candidates)),
        f"candidate_windows={len(candidates)}; clustered_windows={int(clusters['window_count'].sum()) if not clusters.empty else 0}",
        "every detected candidate is assigned to a session/time cluster",
    )
    add(
        "behavior_lfp_covariate_summary_written",
        not behavior_lfp.empty,
        int(behavior_lfp["feature_available"].map(_as_bool).sum()) if not behavior_lfp.empty else 0,
        "numeric behavior/LFP summaries are written when columns are available and missing columns are reported",
    )
    add(
        "trajectory_candidates_detected",
        len(candidates) > 0,
        int(len(candidates)),
        "one or more off-SWR trajectory-family candidates were detected",
        required_for_overall=False,
    )

    required_rows = [row for row in rows if row["required_for_overall"]]
    rows.append(
        {
            "gate": "overall",
            "passed": all(bool(row["passed"]) for row in required_rows),
            "observed": f"{sum(bool(row['passed']) for row in required_rows)}/{len(required_rows)} required gates passed",
            "criterion": "all required off-SWR discovery infrastructure gates pass",
            "required_for_overall": True,
        }
    )
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def off_swr_candidate_specificity_gate_summary(
    candidate_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    candidate_vs_swr: pd.DataFrame,
    candidate_vs_swr_window_table: pd.DataFrame | None = None,
    candidate_vs_swr_model_distribution: pd.DataFrame | None = None,
    run_state_stratified: pd.DataFrame | None = None,
    run_state_specificity: pd.DataFrame | None = None,
    nearest_swr_exclusion: pd.DataFrame | None = None,
    nearest_swr_specificity: pd.DataFrame | None = None,
    candidate_tier_threshold_summary: pd.DataFrame | None = None,
    candidate_tier_session_summary: pd.DataFrame | None = None,
    candidate_tier_rat_summary: pd.DataFrame | None = None,
    candidate_tier_nearest_swr_exclusion: pd.DataFrame | None = None,
    high_specificity_candidates: pd.DataFrame | None = None,
    promotion_readiness: pd.DataFrame | None = None,
    speed_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str, required_for_overall: bool = True) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "required_for_overall": bool(required_for_overall),
            }
        )

    required_columns = set(CANDIDATE_TABLE_COLUMNS)
    present_columns = set(candidate_table.columns)
    missing_columns = sorted(required_columns.difference(present_columns))
    add(
        "candidate_triage_columns_present",
        not missing_columns,
        "missing=" + " ".join(missing_columns) if missing_columns else len(required_columns),
        "ranked candidate table contains the requested triage columns",
    )
    add(
        "candidate_table_ranked",
        candidate_table.empty or candidate_table["candidate_rank"].is_monotonic_increasing,
        int(len(candidate_table)),
        "candidate table has deterministic candidate_rank ordering",
    )
    missing_cluster_ids = (
        int(candidate_table["candidate_cluster_id"].astype(str).eq("").sum())
        if not candidate_table.empty and "candidate_cluster_id" in candidate_table
        else 0
    )
    add(
        "candidate_cluster_ids_complete",
        missing_cluster_ids == 0,
        missing_cluster_ids,
        "every candidate has a candidate_cluster_id",
    )
    add(
        "cluster_table_covers_candidates",
        candidate_table.empty
        or (
            not cluster_table.empty
            and int(cluster_table["window_count"].sum()) == len(candidate_table)
        ),
        f"candidate_windows={len(candidate_table)}; clustered_windows={int(cluster_table['window_count'].sum()) if not cluster_table.empty else 0}",
        "candidate cluster table accounts for every candidate window",
    )
    distance_available = int(_numeric_series(candidate_table, "distance_to_nearest_swr_s").notna().sum()) if not candidate_table.empty else 0
    add(
        "nearest_swr_distance_evaluated",
        candidate_table.empty or distance_available == len(candidate_table),
        f"{distance_available}/{len(candidate_table)}",
        "distance_to_nearest_swr_s is populated for every candidate when real SWR reference windows are present",
    )
    add(
        "candidate_vs_swr_summary_written",
        candidate_table.empty or not candidate_vs_swr.empty,
        int(len(candidate_vs_swr)),
        "candidate-vs-SWR phenotype summary is written when candidates are present",
    )
    contrast_table = candidate_vs_swr_window_table if candidate_vs_swr_window_table is not None else pd.DataFrame()
    contrast_window_sets = set(contrast_table["window_set"].astype(str)) if not contrast_table.empty and "window_set" in contrast_table else set()
    add(
        "candidate_vs_swr_window_table_written",
        candidate_table.empty or "off_swr_candidate" in contrast_window_sets,
        int(len(contrast_table)),
        "direct candidate-vs-SWR window contrast table is written when candidates are present",
    )
    add(
        "swr_reference_windows_available_for_contrast",
        candidate_table.empty or "swr_replay" in contrast_window_sets,
        " ".join(sorted(contrast_window_sets)),
        "direct contrast includes scored real SWR replay reference windows",
    )
    model_distribution = candidate_vs_swr_model_distribution if candidate_vs_swr_model_distribution is not None else pd.DataFrame()
    add(
        "candidate_vs_swr_model_distribution_written",
        candidate_table.empty or not model_distribution.empty,
        int(len(model_distribution)),
        "best trajectory model distributions are written for SWR and off-SWR windows",
    )
    if not candidate_vs_swr.empty and "claim_should_narrow" in candidate_vs_swr:
        should_narrow = bool(candidate_vs_swr["claim_should_narrow"].map(_as_bool).any())
        interpretation = str(_first_value(candidate_vs_swr, "off_swr_vs_swr_interpretation"))
    else:
        should_narrow = False
        interpretation = ""
    add(
        "movement_like_claim_narrowing_flagged",
        True,
        f"claim_should_narrow={should_narrow}; interpretation={interpretation}",
        "summary explicitly flags when off-SWR candidates are mostly high-speed/movement-like",
        required_for_overall=False,
    )
    run_state_summary = run_state_stratified if run_state_stratified is not None else pd.DataFrame()
    run_state_strata = set(run_state_summary["stratum"].astype(str)) if not run_state_summary.empty and "stratum" in run_state_summary else set()
    expected_run_state_strata = {
        "off_swr_immobile_windows",
        "off_swr_running_windows",
        "off_swr_unknown_speed_windows",
        "swr_replay_windows",
    }
    add(
        "run_state_stratified_summary_written",
        expected_run_state_strata.issubset(run_state_strata),
        " ".join(sorted(run_state_strata)),
        "stratified summary reports off-SWR immobile, off-SWR running, unknown-speed, and SWR replay windows",
    )
    run_state_specificity_summary = run_state_specificity if run_state_specificity is not None else pd.DataFrame()
    add(
        "run_state_specificity_summary_written",
        not run_state_specificity_summary.empty,
        int(len(run_state_specificity_summary)),
        "one-row run-state specificity interpretation is written",
    )
    if not run_state_specificity_summary.empty:
        immobile_present = bool(run_state_specificity_summary["immobile_candidate_signal_present"].map(_as_bool).any())
        run_state_narrow = bool(run_state_specificity_summary["claim_should_narrow_for_run_state"].map(_as_bool).any())
        run_state_interpretation = str(_first_value(run_state_specificity_summary, "run_state_specificity_interpretation"))
    else:
        immobile_present = False
        run_state_narrow = False
        run_state_interpretation = ""
    add(
        "immobile_off_swr_candidate_signal_reported",
        True,
        f"immobile_present={immobile_present}; claim_should_narrow={run_state_narrow}; interpretation={run_state_interpretation}",
        "summary reports whether trajectory-family candidates remain present in immobile non-SWR windows",
        required_for_overall=False,
    )
    nearest_exclusion = nearest_swr_exclusion if nearest_swr_exclusion is not None else pd.DataFrame()
    expected_radii = set(DEFAULT_NEAREST_SWR_EXCLUSION_RADII_S)
    observed_radii = (
        set(round(float(value), 6) for value in _numeric_series(nearest_exclusion, "exclusion_radius_s").dropna())
        if not nearest_exclusion.empty
        else set()
    )
    add(
        "nearest_swr_exclusion_summary_written",
        expected_radii.issubset(observed_radii),
        " ".join(str(radius) for radius in sorted(observed_radii)),
        "nearest-SWR exclusion summary reports 100 ms, 250 ms, 500 ms, and 1 s exclusion radii",
    )
    nearest_specificity = nearest_swr_specificity if nearest_swr_specificity is not None else pd.DataFrame()
    add(
        "nearest_swr_specificity_summary_written",
        not nearest_specificity.empty,
        int(len(nearest_specificity)),
        "one-row nearest-SWR specificity interpretation is written",
    )
    if not nearest_specificity.empty:
        nearest_narrow = bool(nearest_specificity["claim_should_narrow_for_nearest_swr"].map(_as_bool).any())
        nearest_interpretation = str(_first_value(nearest_specificity, "nearest_swr_specificity_interpretation"))
    else:
        nearest_narrow = False
        nearest_interpretation = ""
    add(
        "nearest_swr_distance_specificity_reported",
        True,
        f"claim_should_narrow={nearest_narrow}; interpretation={nearest_interpretation}",
        "summary reports whether candidates persist after excluding windows near known SWRs",
        required_for_overall=False,
    )
    expected_tiers = {tier for tier, _ in DEFAULT_CANDIDATE_TIER_THRESHOLDS}
    tier_summary = candidate_tier_threshold_summary if candidate_tier_threshold_summary is not None else pd.DataFrame()
    observed_tiers = set(tier_summary["candidate_tier"].astype(str)) if not tier_summary.empty and "candidate_tier" in tier_summary else set()
    add(
        "candidate_tier_threshold_summary_written",
        expected_tiers.issubset(observed_tiers),
        " ".join(sorted(observed_tiers)),
        "candidate tier threshold summary reports weak, moderate, strong, and extreme thresholds",
    )
    session_tiers = candidate_tier_session_summary if candidate_tier_session_summary is not None else pd.DataFrame()
    rat_tiers = candidate_tier_rat_summary if candidate_tier_rat_summary is not None else pd.DataFrame()
    add(
        "candidate_tier_rat_session_summaries_written",
        (not session_tiers.empty) and (not rat_tiers.empty),
        f"session_rows={len(session_tiers)}; rat_rows={len(rat_tiers)}",
        "candidate tier summaries report counts by session and rat",
    )
    tier_distance = candidate_tier_nearest_swr_exclusion if candidate_tier_nearest_swr_exclusion is not None else pd.DataFrame()
    observed_tier_distance_tiers = (
        set(tier_distance["candidate_tier"].astype(str)) if not tier_distance.empty and "candidate_tier" in tier_distance else set()
    )
    observed_tier_distance_radii = (
        set(round(float(value), 6) for value in _numeric_series(tier_distance, "exclusion_radius_s").dropna())
        if not tier_distance.empty
        else set()
    )
    add(
        "candidate_tier_nearest_swr_exclusion_summary_written",
        expected_tiers.issubset(observed_tier_distance_tiers)
        and set(DEFAULT_NEAREST_SWR_EXCLUSION_RADII_S).issubset(observed_tier_distance_radii),
        f"tiers={' '.join(sorted(observed_tier_distance_tiers))}; radii={' '.join(str(radius) for radius in sorted(observed_tier_distance_radii))}",
        "candidate tier nearest-SWR exclusion summary reports all tiers across all exclusion radii",
    )
    high_specificity = high_specificity_candidates if high_specificity_candidates is not None else pd.DataFrame()
    add(
        "high_specificity_candidate_table_written",
        candidate_table.empty or set(HIGH_SPECIFICITY_CANDIDATE_COLUMNS).issubset(high_specificity.columns),
        int(len(high_specificity)),
        "high-specificity candidate table reports strong-tier candidates surviving the 1 s nearest-SWR exclusion",
    )
    readiness = promotion_readiness if promotion_readiness is not None else pd.DataFrame()
    add(
        "promotion_readiness_summary_written",
        not readiness.empty,
        int(len(readiness)),
        "one-row off-SWR replay promotion readiness summary is written",
    )
    if not readiness.empty:
        promotion_status = str(_first_value(readiness, "promotion_status"))
        promotion_ready = bool(readiness["promotion_ready"].map(_as_bool).any())
    else:
        promotion_status = ""
        promotion_ready = False
    add(
        "off_swr_replay_promotion_status_reported",
        True,
        f"promotion_ready={promotion_ready}; status={promotion_status}",
        "summary reports whether off-SWR candidates are promotable or remain exploratory",
        required_for_overall=False,
    )
    coverage = speed_coverage if speed_coverage is not None else pd.DataFrame()
    add(
        "speed_coverage_summary_written",
        not coverage.empty,
        int(len(coverage)),
        "one-row speed metadata coverage summary is written",
    )
    if not coverage.empty:
        speed_ready = bool(coverage["speed_coverage_ready"].map(_as_bool).any())
        speed_status = str(_first_value(coverage, "speed_coverage_status"))
    else:
        speed_ready = False
        speed_status = ""
    add(
        "speed_metadata_coverage_reported",
        True,
        f"speed_coverage_ready={speed_ready}; status={speed_status}",
        "summary reports whether scored off-SWR artifacts contain speed metadata for promotion gating",
        required_for_overall=False,
    )
    speed_available = int(_numeric_series(candidate_table, "animal_speed_mean").notna().sum()) if not candidate_table.empty else 0
    add(
        "animal_speed_available",
        speed_available > 0,
        f"{speed_available}/{len(candidate_table)}",
        "animal-speed phenotype columns are populated for at least one candidate",
        required_for_overall=False,
    )
    entropy_available = int(_numeric_series(candidate_table, "trajectory_posterior_entropy").notna().sum()) if not candidate_table.empty else 0
    add(
        "trajectory_entropy_available",
        entropy_available > 0,
        f"{entropy_available}/{len(candidate_table)}",
        "trajectory posterior entropy is populated for at least one candidate",
        required_for_overall=False,
    )
    movement_like = (
        int(candidate_table["candidate_specificity_label"].astype(str).eq(MOVEMENT_SPIKING_LIKE_LABEL).sum())
        if not candidate_table.empty
        else 0
    )
    add(
        "movement_spiking_like_candidates_flagged",
        movement_like >= 0,
        movement_like,
        "candidate table explicitly flags ordinary movement/spiking-like candidates",
        required_for_overall=False,
    )

    required_rows = [row for row in rows if row["required_for_overall"]]
    rows.append(
        {
            "gate": "overall",
            "passed": all(bool(row["passed"]) for row in required_rows),
            "observed": f"{sum(bool(row['passed']) for row in required_rows)}/{len(required_rows)} required gates passed",
            "criterion": "all required off-SWR candidate triage/specificity gates pass",
            "required_for_overall": True,
        }
    )
    return pd.DataFrame(rows, columns=list(GATE_COLUMNS))


def write_off_swr_trajectory_discovery_outputs(
    scores: pd.DataFrame,
    output: str | Path,
    *,
    comparison_scope: str = "full-core",
    required_models: tuple[str, ...] | None = None,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    cluster_gap_s: float = 0.5,
    run_speed_threshold_cm_s: float = 5.0,
    behavior_lfp_columns: Sequence[str] = DEFAULT_BEHAVIOR_LFP_COLUMNS,
) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    scored_models = tuple(scores["model"].dropna().astype(str).unique()) if "model" in scores.columns else tuple()
    resolved_required_models, resolved_trajectory_models = resolve_family_model_sets(
        comparison_scope=comparison_scope,
        scored_models=scored_models,
    )
    effective_required_models = tuple(required_models or resolved_required_models)
    effective_trajectory_models = (
        tuple(model for model in resolved_trajectory_models if model in set(effective_required_models))
        if required_models is None
        else tuple(model for model in effective_required_models if not str(model).endswith("stationary"))
    )
    decisions = off_swr_trajectory_decisions(
        scores,
        comparison_scope=comparison_scope,
        required_models=effective_required_models,
        margin_threshold=margin_threshold,
        behavior_lfp_columns=behavior_lfp_columns,
    )
    candidates = off_swr_trajectory_candidates(decisions)
    clusters = cluster_off_swr_candidates(candidates, cluster_gap_s=cluster_gap_s)
    candidate_table = off_swr_candidate_table(
        decisions,
        scores,
        required_models=effective_required_models,
        trajectory_models=effective_trajectory_models,
        cluster_gap_s=cluster_gap_s,
        run_speed_threshold_cm_s=run_speed_threshold_cm_s,
    )
    candidate_cluster_table = off_swr_candidate_cluster_table(candidate_table)
    candidate_session_summary = off_swr_candidate_group_summary(candidate_table, ("session",))
    candidate_rat_summary = off_swr_candidate_group_summary(candidate_table, ("rat",))
    candidate_vs_swr_window_table = off_swr_candidate_vs_swr_window_table(
        candidate_table,
        scores,
        comparison_scope=comparison_scope,
        required_models=effective_required_models,
        margin_threshold=margin_threshold,
        run_speed_threshold_cm_s=run_speed_threshold_cm_s,
    )
    candidate_vs_swr_model_distribution = off_swr_candidate_vs_swr_model_distribution(candidate_vs_swr_window_table)
    candidate_vs_swr = off_swr_candidate_vs_swr_summary(
        candidate_table,
        candidate_vs_swr_window_table,
        candidate_vs_swr_model_distribution,
    )
    off_swr_phenotype_windows = _off_swr_run_state_window_table(
        decisions,
        scores,
        required_models=effective_required_models,
        trajectory_models=effective_trajectory_models,
        run_speed_threshold_cm_s=run_speed_threshold_cm_s,
    )
    run_state_stratified = off_swr_run_state_stratified_summary(
        decisions,
        scores,
        candidate_vs_swr_window_table,
        required_models=effective_required_models,
        trajectory_models=effective_trajectory_models,
        margin_threshold=margin_threshold,
        run_speed_threshold_cm_s=run_speed_threshold_cm_s,
        off_swr_windows=off_swr_phenotype_windows,
    )
    run_state_specificity = off_swr_run_state_specificity_summary(run_state_stratified)
    nearest_swr_exclusion = off_swr_nearest_swr_exclusion_summary(
        decisions,
        scores,
        required_models=effective_required_models,
        trajectory_models=effective_trajectory_models,
        exclusion_radii_s=DEFAULT_NEAREST_SWR_EXCLUSION_RADII_S,
        run_speed_threshold_cm_s=run_speed_threshold_cm_s,
        off_swr_windows=off_swr_phenotype_windows,
    )
    nearest_swr_specificity = off_swr_nearest_swr_specificity_summary(nearest_swr_exclusion)
    candidate_tier_threshold_summary = off_swr_candidate_tier_threshold_summary(off_swr_phenotype_windows)
    candidate_tier_session_summary = off_swr_candidate_tier_group_summary(off_swr_phenotype_windows, ("session",))
    candidate_tier_rat_summary = off_swr_candidate_tier_group_summary(off_swr_phenotype_windows, ("rat",))
    candidate_tier_nearest_swr_exclusion = off_swr_candidate_tier_nearest_swr_exclusion_summary(off_swr_phenotype_windows)
    high_specificity_candidates = off_swr_high_specificity_candidate_table(candidate_table)
    promotion_readiness = off_swr_promotion_readiness_summary(
        candidate_table,
        high_specificity_candidates,
        nearest_swr_specificity,
        run_state_specificity,
    )
    speed_coverage = off_swr_speed_coverage_summary(
        off_swr_phenotype_windows,
        candidate_table,
        candidate_vs_swr_window_table,
        promotion_readiness,
    )
    behavior_lfp = off_swr_behavior_lfp_summary(decisions, behavior_lfp_columns=behavior_lfp_columns)
    outputs = {
        "off_swr_trajectory_discovery_event_model_evidence.csv": scores,
        "off_swr_trajectory_discovery_decisions.csv": decisions,
        "off_swr_trajectory_candidate_events.csv": candidates,
        "off_swr_trajectory_candidate_summary.csv": off_swr_candidate_summary(decisions),
        "session_off_swr_candidate_summary.csv": off_swr_group_summary(decisions, ("session",)),
        "rat_off_swr_candidate_summary.csv": off_swr_group_summary(decisions, ("rat",)),
        "off_swr_candidate_clusters.csv": clusters,
        "off_swr_candidate_behavior_lfp_summary.csv": behavior_lfp,
        "off_swr_candidate_gate_summary.csv": off_swr_discovery_gate_summary(decisions, candidates, clusters, behavior_lfp),
        "off_swr_candidate_table.csv": candidate_table,
        "off_swr_candidate_cluster_table.csv": candidate_cluster_table,
        "off_swr_candidate_session_summary.csv": candidate_session_summary,
        "off_swr_candidate_rat_summary.csv": candidate_rat_summary,
        "off_swr_candidate_vs_swr_summary.csv": candidate_vs_swr,
        "off_swr_candidate_vs_swr_window_table.csv": candidate_vs_swr_window_table,
        "off_swr_candidate_vs_swr_model_distribution.csv": candidate_vs_swr_model_distribution,
        "off_swr_run_state_stratified_summary.csv": run_state_stratified,
        "off_swr_run_state_specificity_summary.csv": run_state_specificity,
        "off_swr_nearest_swr_exclusion_summary.csv": nearest_swr_exclusion,
        "off_swr_nearest_swr_specificity_summary.csv": nearest_swr_specificity,
        "off_swr_candidate_tier_threshold_summary.csv": candidate_tier_threshold_summary,
        "off_swr_candidate_tier_session_summary.csv": candidate_tier_session_summary,
        "off_swr_candidate_tier_rat_summary.csv": candidate_tier_rat_summary,
        "off_swr_candidate_tier_nearest_swr_exclusion_summary.csv": candidate_tier_nearest_swr_exclusion,
        "off_swr_high_specificity_candidate_table.csv": high_specificity_candidates,
        "off_swr_promotion_readiness_summary.csv": promotion_readiness,
        "off_swr_speed_coverage_summary.csv": speed_coverage,
        "off_swr_candidate_specificity_gate_summary.csv": off_swr_candidate_specificity_gate_summary(
            candidate_table,
            candidate_cluster_table,
            candidate_vs_swr,
            candidate_vs_swr_window_table,
            candidate_vs_swr_model_distribution,
            run_state_stratified,
            run_state_specificity,
            nearest_swr_exclusion,
            nearest_swr_specificity,
            candidate_tier_threshold_summary,
            candidate_tier_session_summary,
            candidate_tier_rat_summary,
            candidate_tier_nearest_swr_exclusion,
            high_specificity_candidates,
            promotion_readiness,
            speed_coverage,
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-glob", required=True)
    parser.add_argument("--output", default="results/off-swr-trajectory-discovery")
    parser.add_argument(
        "--comparison-scope",
        default="full-core",
        choices=("auto", "full-core", LIGHTWEIGHT_FO_IMM_STATIONARY_SCOPE),
    )
    parser.add_argument(
        "--required-models",
        default="",
        help="Optional whitespace-separated required models. Defaults to the full-core exact discovery set.",
    )
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)
    parser.add_argument("--cluster-gap-s", type=float, default=0.5)
    parser.add_argument(
        "--run-speed-threshold-cm-s",
        type=float,
        default=5.0,
        help="Speed threshold used to label candidate windows as run versus immobility/unknown when speed columns are present.",
    )
    parser.add_argument(
        "--behavior-lfp-columns",
        default=" ".join(DEFAULT_BEHAVIOR_LFP_COLUMNS),
        help="Whitespace/comma-separated behavior or LFP columns to summarize if present.",
    )
    args = parser.parse_args()

    required_models = _parse_names(args.required_models, FULL_CORE_REQUIRED_MODELS) if args.required_models.strip() else None
    behavior_lfp_columns = _parse_names(args.behavior_lfp_columns, DEFAULT_BEHAVIOR_LFP_COLUMNS)
    scores = _read_score_files(args.score_glob)
    outputs = write_off_swr_trajectory_discovery_outputs(
        scores,
        args.output,
        comparison_scope=args.comparison_scope,
        required_models=required_models,
        margin_threshold=args.margin_threshold,
        cluster_gap_s=args.cluster_gap_s,
        run_speed_threshold_cm_s=args.run_speed_threshold_cm_s,
        behavior_lfp_columns=behavior_lfp_columns,
    )
    print("Off-SWR trajectory discovery summary:")
    print(outputs["off_swr_trajectory_candidate_summary.csv"].to_string(index=False))
    print("\nOff-SWR trajectory discovery gates:")
    print(outputs["off_swr_candidate_gate_summary.csv"].to_string(index=False))
    print("\nOff-SWR candidate specificity gates:")
    print(outputs["off_swr_candidate_specificity_gate_summary.csv"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
