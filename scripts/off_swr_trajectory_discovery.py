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
)


DEFAULT_OFF_SWR_DISCOVERY_MODELS = (
    "sorted-spike-state-space-stationary",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
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

KEY_COLUMNS = ("session", "event_index", "window_role", "null_index")
TRAJECTORY_CANDIDATE_CLASS = "off_swr_trajectory_family_candidate"
STATIC_NONTRAJECTORY_CLASS = "off_swr_static_nontrajectory"
AMBIGUOUS_CLASS = "ambiguous"
INCOMPLETE_CLASS = "incomplete_core"
EXCLUDED_SWR_OVERLAP_CLASS = "excluded_known_swr_overlap"

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


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


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


def write_off_swr_trajectory_discovery_outputs(
    scores: pd.DataFrame,
    output: str | Path,
    *,
    comparison_scope: str = "full-core",
    required_models: tuple[str, ...] | None = None,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    cluster_gap_s: float = 0.5,
    behavior_lfp_columns: Sequence[str] = DEFAULT_BEHAVIOR_LFP_COLUMNS,
) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    decisions = off_swr_trajectory_decisions(
        scores,
        comparison_scope=comparison_scope,
        required_models=required_models,
        margin_threshold=margin_threshold,
        behavior_lfp_columns=behavior_lfp_columns,
    )
    candidates = off_swr_trajectory_candidates(decisions)
    clusters = cluster_off_swr_candidates(candidates, cluster_gap_s=cluster_gap_s)
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
        behavior_lfp_columns=behavior_lfp_columns,
    )
    print("Off-SWR trajectory discovery summary:")
    print(outputs["off_swr_trajectory_candidate_summary.csv"].to_string(index=False))
    print("\nOff-SWR trajectory discovery gates:")
    print(outputs["off_swr_candidate_gate_summary.csv"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
