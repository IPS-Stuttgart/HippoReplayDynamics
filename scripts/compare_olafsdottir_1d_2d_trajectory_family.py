#!/usr/bin/env python3
"""Compare Olafsdottir 1D Z-track and Pfeiffer/Foster 2D replay evidence.

The comparison is a summary layer over completed evidence artifacts. It is not
an event scorer. Use event-level model-evidence tables so raw family margins can
be normalized per spike and per time bin before interpreting 1D-vs-2D patterns.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EXACT_EVIDENCE_SUPPORT,
    _coerce_bool_series,
    ensure_evidence_support_columns,
)


STATIONARY_MODEL = "sorted-spike-state-space-stationary"
DIFFUSION_MODEL = "sorted-spike-state-space-diffusion"
FRAGMENTED_MODEL = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM_MODEL = "sorted-spike-state-space-first-order-imm"
MOMENTUM_MODEL = "sorted-spike-state-space-momentum-exact-sparse"
EXACT_CORE_MODELS: tuple[str, ...] = (
    STATIONARY_MODEL,
    DIFFUSION_MODEL,
    FRAGMENTED_MODEL,
    FIRST_ORDER_IMM_MODEL,
    MOMENTUM_MODEL,
)
TRAJECTORY_MODELS: tuple[str, ...] = (
    DIFFUSION_MODEL,
    FRAGMENTED_MODEL,
    FIRST_ORDER_IMM_MODEL,
    MOMENTUM_MODEL,
)
EVENT_TABLE_CANDIDATES = (
    "olafsdottir_1d_event_model_evidence.csv",
    "all_sessions_event_model_evidence.csv",
    "event_model_evidence.csv",
    "event_model_evidence_with_margins.csv",
)
SUMMARY_OUTPUT = "compare_1d_2d_trajectory_family_summary.csv"
INTERPRETATION_OUTPUT = "compare_1d_2d_interpretation_summary.csv"
READINESS_OUTPUT = "compare_1d_2d_biological_readiness_gates.csv"
PRIMARY_COLUMNS = (
    "dataset",
    "environment_type",
    "events",
    "trajectory_confident_claim_fraction",
    "nontrajectory_confident_claim_fraction",
    "mean_family_margin",
    "median_family_margin",
    "first_order_imm_raw_best_fraction",
    "momentum_raw_best_fraction",
    "momentum_vs_diffusion_median",
    "mean_family_margin_per_spike",
    "median_family_margin_per_spike",
    "mean_family_margin_per_time_bin",
    "median_family_margin_per_time_bin",
    "mean_spikes_per_event",
    "median_spikes_per_event",
    "mean_time_bins_per_event",
    "median_time_bins_per_event",
)


def resolve_event_table(path: str | Path) -> Path:
    """Resolve a CSV path or artifact directory to an event-level table."""

    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Evidence artifact path does not exist: {candidate}")
    for name in EVENT_TABLE_CANDIDATES:
        table = candidate / name
        if table.is_file():
            return table
    nested = sorted(candidate.rglob("all_sessions_event_model_evidence.csv"))
    nested.extend(sorted(candidate.rglob("olafsdottir_1d_event_model_evidence.csv")))
    nested.extend(sorted(candidate.rglob("event_model_evidence.csv")))
    if nested:
        return nested[0]
    raise FileNotFoundError(f"No event model-evidence CSV found under {candidate}")


def load_event_scores(path: str | Path) -> pd.DataFrame:
    table = resolve_event_table(path)
    frame = pd.read_csv(table)
    source = pd.Series(str(table), index=frame.index, name="source_event_table")
    return pd.concat([frame.copy(), source], axis=1)


def build_comparison(
    *,
    one_d_scores: pd.DataFrame,
    two_d_scores: pd.DataFrame,
    output: str | Path,
    margin_threshold: float = 5.5,
    one_d_dataset: str = "Olafsdottir2016",
    two_d_dataset: str = "PfeifferFoster",
    one_d_environment: str = "1D_Z_track",
    two_d_environment: str = "2D_open_field",
    min_robust_1d_events: int = 50,
    min_1d_animals: int = 2,
    min_1d_sessions: int = 2,
    weaker_fraction_delta: float = 0.20,
    similar_fraction_delta: float = 0.10,
    cell_identity_verified: bool = False,
    synthetic_1d_tests_passed: bool = False,
    linearization_diagnostics: str | Path | pd.DataFrame | None = None,
    event_detection_summary: str | Path | pd.DataFrame | None = None,
    min_linearization_valid_fraction: float = 0.90,
    max_linearization_median_projection_error_cm: float = 15.0,
    min_event_candidates: int = 10,
    min_event_median_spikes: float = 5.0,
) -> dict[str, pd.DataFrame]:
    """Write comparison and interpretation CSVs."""

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    one_d_event_metrics = event_level_metrics(one_d_scores, margin_threshold=margin_threshold)
    two_d_event_metrics = event_level_metrics(two_d_scores, margin_threshold=margin_threshold)
    summary = pd.DataFrame(
        [
            summarize_dataset(
                one_d_event_metrics,
                dataset=one_d_dataset,
                environment_type=one_d_environment,
                margin_threshold=margin_threshold,
            ),
            summarize_dataset(
                two_d_event_metrics,
                dataset=two_d_dataset,
                environment_type=two_d_environment,
                margin_threshold=margin_threshold,
            ),
        ]
    )
    summary = summary[[*PRIMARY_COLUMNS, *[column for column in summary.columns if column not in PRIMARY_COLUMNS]]]
    readiness = biological_readiness_gates(
        one_d_scores=one_d_scores,
        one_d_event_metrics=one_d_event_metrics,
        comparison_summary=summary,
        min_1d_animals=min_1d_animals,
        min_1d_sessions=min_1d_sessions,
        min_robust_1d_events=min_robust_1d_events,
        cell_identity_verified=cell_identity_verified,
        synthetic_1d_tests_passed=synthetic_1d_tests_passed,
        linearization_diagnostics=linearization_diagnostics,
        event_detection_summary=event_detection_summary,
        min_linearization_valid_fraction=min_linearization_valid_fraction,
        max_linearization_median_projection_error_cm=max_linearization_median_projection_error_cm,
        min_event_candidates=min_event_candidates,
        min_event_median_spikes=min_event_median_spikes,
    )
    failed_readiness_gates = tuple(readiness.loc[~readiness["passed"].map(bool), "gate"].astype(str))
    interpretation = interpretation_summary(
        summary,
        margin_threshold=margin_threshold,
        min_robust_1d_events=min_robust_1d_events,
        weaker_fraction_delta=weaker_fraction_delta,
        similar_fraction_delta=similar_fraction_delta,
        biological_ready=not failed_readiness_gates,
        failed_readiness_gates=failed_readiness_gates,
    )
    summary.to_csv(output_dir / SUMMARY_OUTPUT, index=False)
    interpretation.to_csv(output_dir / INTERPRETATION_OUTPUT, index=False)
    readiness.to_csv(output_dir / READINESS_OUTPUT, index=False)
    return {
        "comparison_summary": summary,
        "interpretation_summary": interpretation,
        "readiness_gates": readiness,
        "one_d_event_metrics": one_d_event_metrics,
        "two_d_event_metrics": two_d_event_metrics,
    }


def event_level_metrics(scores: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    """Return complete exact-core event-level margins and normalized metrics."""

    exact = _exact_core_rows(scores)
    columns = [
        "session",
        "event_index",
        "family_margin",
        "family_margin_per_spike",
        "family_margin_per_time_bin",
        "trajectory_confident_claim",
        "nontrajectory_confident_claim",
        "best_trajectory_model",
        "best_nontrajectory_model",
        "best_core_model",
        "first_order_imm_raw_best",
        "momentum_raw_best",
        "momentum_minus_diffusion",
        "n_spikes",
        "n_time",
        "complete_exact_core",
    ]
    if exact.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for (session, event_index), group in exact.groupby(["session", "event_index"], sort=True):
        models = set(group["model"].astype(str))
        complete = set(EXACT_CORE_MODELS).issubset(models)
        if not complete:
            continue
        by_model = group.set_index("model", drop=False)
        trajectory = by_model.loc[list(TRAJECTORY_MODELS)].sort_values("log_evidence", ascending=False)
        nontrajectory = by_model.loc[[STATIONARY_MODEL]].sort_values("log_evidence", ascending=False)
        core = by_model.loc[list(EXACT_CORE_MODELS)].sort_values("log_evidence", ascending=False)
        best_trajectory = trajectory.iloc[0]
        best_nontrajectory = nontrajectory.iloc[0]
        best_core = core.iloc[0]
        family_margin = float(best_trajectory["log_evidence"]) - float(best_nontrajectory["log_evidence"])
        n_spikes = _event_first_numeric(group, "n_spikes")
        n_time = _event_first_numeric(group, "n_time")
        if not np.isfinite(n_time) or n_time <= 0:
            n_time = _fallback_time_bins(group)
        momentum_minus_diffusion = float(by_model.loc[MOMENTUM_MODEL, "log_evidence"]) - float(by_model.loc[DIFFUSION_MODEL, "log_evidence"])
        rows.append(
            {
                "session": str(session),
                "event_index": int(event_index),
                "family_margin": family_margin,
                "family_margin_per_spike": _safe_divide(family_margin, n_spikes),
                "family_margin_per_time_bin": _safe_divide(family_margin, n_time),
                "trajectory_confident_claim": bool(family_margin >= margin_threshold),
                "nontrajectory_confident_claim": bool(family_margin <= -margin_threshold),
                "best_trajectory_model": str(best_trajectory["model"]),
                "best_nontrajectory_model": str(best_nontrajectory["model"]),
                "best_core_model": str(best_core["model"]),
                "first_order_imm_raw_best": str(best_core["model"]) == FIRST_ORDER_IMM_MODEL,
                "momentum_raw_best": str(best_core["model"]) == MOMENTUM_MODEL,
                "momentum_minus_diffusion": momentum_minus_diffusion,
                "n_spikes": n_spikes,
                "n_time": n_time,
                "complete_exact_core": True,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_dataset(
    events: pd.DataFrame,
    *,
    dataset: str,
    environment_type: str,
    margin_threshold: float,
) -> dict[str, object]:
    event_count = int(len(events))
    row = {
        "dataset": dataset,
        "environment_type": environment_type,
        "events": event_count,
        "trajectory_confident_claim_fraction": _mean_bool(events, "trajectory_confident_claim"),
        "nontrajectory_confident_claim_fraction": _mean_bool(events, "nontrajectory_confident_claim"),
        "mean_family_margin": _numeric_mean(events, "family_margin"),
        "median_family_margin": _numeric_median(events, "family_margin"),
        "first_order_imm_raw_best_fraction": _mean_bool(events, "first_order_imm_raw_best"),
        "momentum_raw_best_fraction": _mean_bool(events, "momentum_raw_best"),
        "momentum_vs_diffusion_median": _numeric_median(events, "momentum_minus_diffusion"),
        "mean_family_margin_per_spike": _numeric_mean(events, "family_margin_per_spike"),
        "median_family_margin_per_spike": _numeric_median(events, "family_margin_per_spike"),
        "mean_family_margin_per_time_bin": _numeric_mean(events, "family_margin_per_time_bin"),
        "median_family_margin_per_time_bin": _numeric_median(events, "family_margin_per_time_bin"),
        "mean_spikes_per_event": _numeric_mean(events, "n_spikes"),
        "median_spikes_per_event": _numeric_median(events, "n_spikes"),
        "mean_time_bins_per_event": _numeric_mean(events, "n_time"),
        "median_time_bins_per_event": _numeric_median(events, "n_time"),
        "margin_threshold": float(margin_threshold),
        "complete_exact_core_events": event_count,
    }
    return row


def interpretation_summary(
    summary: pd.DataFrame,
    *,
    margin_threshold: float,
    min_robust_1d_events: int,
    weaker_fraction_delta: float,
    similar_fraction_delta: float,
    biological_ready: bool = True,
    failed_readiness_gates: Sequence[str] = (),
) -> pd.DataFrame:
    columns = [
        "comparison",
        "interpretation_class",
        "directional_pattern",
        "claim_strength",
        "one_d_events",
        "two_d_events",
        "trajectory_confident_claim_fraction_delta_1d_minus_2d",
        "median_family_margin_delta_1d_minus_2d",
        "median_family_margin_per_spike_delta_1d_minus_2d",
        "median_family_margin_per_time_bin_delta_1d_minus_2d",
        "first_order_imm_raw_best_fraction_delta_1d_minus_2d",
        "momentum_raw_best_fraction_delta_1d_minus_2d",
        "paper_safe_statement",
        "hard_caveat",
        "biological_readiness_status",
        "failed_readiness_gates",
        "margin_threshold",
        "min_robust_1d_events",
    ]
    if summary.empty or len(summary) < 2:
        return pd.DataFrame(
            [
                {
                    "comparison": "1D_Z_track_vs_2D_open_field",
                    "interpretation_class": "missing_inputs",
                    "directional_pattern": "missing_inputs",
                    "claim_strength": "not_interpretable",
                    "one_d_events": 0,
                    "two_d_events": 0,
                    "trajectory_confident_claim_fraction_delta_1d_minus_2d": np.nan,
                    "median_family_margin_delta_1d_minus_2d": np.nan,
                    "median_family_margin_per_spike_delta_1d_minus_2d": np.nan,
                    "median_family_margin_per_time_bin_delta_1d_minus_2d": np.nan,
                    "first_order_imm_raw_best_fraction_delta_1d_minus_2d": np.nan,
                    "momentum_raw_best_fraction_delta_1d_minus_2d": np.nan,
                    "paper_safe_statement": "The 1D-vs-2D comparison inputs are incomplete.",
                    "hard_caveat": _hard_caveat(),
                    "biological_readiness_status": "not_ready",
                    "failed_readiness_gates": "missing_inputs",
                    "margin_threshold": float(margin_threshold),
                    "min_robust_1d_events": int(min_robust_1d_events),
                }
            ],
            columns=columns,
        )

    one_d = summary[summary["environment_type"].astype(str).str.startswith("1D")]
    two_d = summary[summary["environment_type"].astype(str).str.startswith("2D")]
    if one_d.empty or two_d.empty:
        one_d = summary.iloc[[0]]
        two_d = summary.iloc[[1]]
    else:
        one_d = one_d.iloc[[0]]
        two_d = two_d.iloc[[0]]
    one = one_d.iloc[0]
    two = two_d.iloc[0]
    deltas = {
        "trajectory_confident_claim_fraction_delta_1d_minus_2d": _delta(one, two, "trajectory_confident_claim_fraction"),
        "median_family_margin_delta_1d_minus_2d": _delta(one, two, "median_family_margin"),
        "median_family_margin_per_spike_delta_1d_minus_2d": _delta(one, two, "median_family_margin_per_spike"),
        "median_family_margin_per_time_bin_delta_1d_minus_2d": _delta(one, two, "median_family_margin_per_time_bin"),
        "first_order_imm_raw_best_fraction_delta_1d_minus_2d": _delta(one, two, "first_order_imm_raw_best_fraction"),
        "momentum_raw_best_fraction_delta_1d_minus_2d": _delta(one, two, "momentum_raw_best_fraction"),
    }
    one_events = int(one["events"])
    two_events = int(two["events"])
    data_limited = one_events < int(min_robust_1d_events)
    directional = _directional_pattern(
        one,
        two,
        weaker_fraction_delta=weaker_fraction_delta,
        similar_fraction_delta=similar_fraction_delta,
    )
    if not biological_ready:
        interpretation_class = "biological_comparison_not_ready"
        claim_strength = "pre_biological_comparison_not_ready"
        statement = (
            "Do not use the directional 1D-vs-2D pattern as a biological comparison "
            "until all readiness gates pass."
        )
    elif data_limited:
        interpretation_class = "sparse_or_data_limited_feasibility_result"
        claim_strength = "hypothesis_generating_smoke"
        statement = (
            "The current 1D pilot is data-limited; directional differences can guide scaling, "
            "but they are not a biological conclusion."
        )
    else:
        interpretation_class = directional
        claim_strength = "robust_comparison_candidate"
        statement = _statement_for_directional_pattern(directional)
    row = {
        "comparison": "1D_Z_track_vs_2D_open_field",
        "interpretation_class": interpretation_class,
        "directional_pattern": directional,
        "claim_strength": claim_strength,
        "one_d_events": one_events,
        "two_d_events": two_events,
        **deltas,
        "paper_safe_statement": statement,
        "hard_caveat": _hard_caveat(),
        "biological_readiness_status": "ready" if biological_ready else "not_ready",
        "failed_readiness_gates": " ".join(failed_readiness_gates),
        "margin_threshold": float(margin_threshold),
        "min_robust_1d_events": int(min_robust_1d_events),
    }
    return pd.DataFrame([row], columns=columns)


def biological_readiness_gates(
    *,
    one_d_scores: pd.DataFrame,
    one_d_event_metrics: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    min_1d_animals: int,
    min_1d_sessions: int,
    min_robust_1d_events: int,
    cell_identity_verified: bool,
    synthetic_1d_tests_passed: bool,
    linearization_diagnostics: str | Path | pd.DataFrame | None,
    event_detection_summary: str | Path | pd.DataFrame | None,
    min_linearization_valid_fraction: float,
    max_linearization_median_projection_error_cm: float,
    min_event_candidates: int,
    min_event_median_spikes: float,
) -> pd.DataFrame:
    """Return gates that must pass before making a biological 1D-vs-2D claim."""

    one_d_summary = _one_d_summary_row(comparison_summary)
    animals = _animal_count(one_d_scores)
    sessions = int(one_d_scores["session"].nunique()) if "session" in one_d_scores else 0
    events = int(one_d_summary.get("events", len(one_d_event_metrics))) if not one_d_summary.empty else int(len(one_d_event_metrics))
    exact_complete = int(one_d_summary.get("complete_exact_core_events", len(one_d_event_metrics))) if not one_d_summary.empty else int(len(one_d_event_metrics))
    linearization = _linearization_diagnostic_values(linearization_diagnostics)
    event_detection = _event_detection_values(event_detection_summary, one_d_event_metrics)
    normalized = _normalized_columns_present(one_d_summary)

    gates = [
        _gate(
            "multiple_animals_sessions",
            animals >= min_1d_animals and sessions >= min_1d_sessions and events >= min_robust_1d_events,
            f"animals={animals}; sessions={sessions}; events={events}",
            f"animals>={min_1d_animals}, sessions>={min_1d_sessions}, events>={min_robust_1d_events}",
            "Biological comparison needs scaled 1D evidence, not a single-animal/day pilot.",
        ),
        _gate(
            "track_sleep_cell_identity_verified",
            bool(cell_identity_verified),
            str(bool(cell_identity_verified)).lower(),
            "explicit verification flag is true",
            "Stable Track1/SleepPOST cell identity cannot be inferred from model-evidence rows alone.",
        ),
        _gate(
            "linearization_diagnostics_acceptable",
            bool(linearization["available"])
            and linearization["fraction_valid_position"] >= min_linearization_valid_fraction
            and linearization["median_projection_error_cm"] <= max_linearization_median_projection_error_cm
            and linearization["track_length_cm"] > 0
            and linearization["occupied_linear_bins"] > 1,
            (
                f"available={linearization['available']}; "
                f"fraction_valid_position={linearization['fraction_valid_position']:.6g}; "
                f"median_projection_error_cm={linearization['median_projection_error_cm']:.6g}; "
                f"track_length_cm={linearization['track_length_cm']:.6g}; "
                f"occupied_linear_bins={linearization['occupied_linear_bins']}"
            ),
            (
                f"available and fraction_valid_position>={min_linearization_valid_fraction:g}, "
                f"median_projection_error_cm<={max_linearization_median_projection_error_cm:g}, "
                "track_length_cm>0, occupied_linear_bins>1"
            ),
            "Use visible linearization diagnostics before interpreting 1D replay geometry.",
        ),
        _gate(
            "event_detection_plausible",
            event_detection["event_candidates"] >= min_event_candidates
            and event_detection["median_event_spikes"] >= min_event_median_spikes,
            (
                f"event_candidates={event_detection['event_candidates']}; "
                f"median_event_spikes={event_detection['median_event_spikes']:.6g}"
            ),
            f"event_candidates>={min_event_candidates}, median_event_spikes>={min_event_median_spikes:g}",
            "Ripple/burst events should be plausible before comparing model hierarchies.",
        ),
        _gate(
            "synthetic_1d_state_space_tests_passed",
            bool(synthetic_1d_tests_passed),
            str(bool(synthetic_1d_tests_passed)).lower(),
            "explicit synthetic-test flag is true",
            "The tiny-grid 1D state-space correctness tests must be run in the validation context.",
        ),
        _gate(
            "exact_core_coverage_complete",
            events > 0 and exact_complete == events,
            f"complete_exact_core_events={exact_complete}; events={events}",
            "complete_exact_core_events == events",
            "Every 1D event needs stationary, diffusion, fragmented, first-order IMM, and exact-sparse momentum rows.",
        ),
        _gate(
            "normalized_margin_columns_present",
            bool(normalized["passed"]),
            normalized["value"],
            "finite per-spike and per-time-bin normalized margin summaries",
            "Raw 1D and 2D log-evidence margins are not directly comparable.",
        ),
        _gate(
            "within_dataset_decisions_only",
            True,
            "implemented_by_construction",
            "model decisions are computed within each dataset before cross-dataset summary comparison",
            "The script compares within-dataset family/model decisions and normalized summaries, not raw cross-dataset logZ.",
        ),
    ]
    return pd.DataFrame(gates)


def _directional_pattern(
    one: pd.Series,
    two: pd.Series,
    *,
    weaker_fraction_delta: float,
    similar_fraction_delta: float,
) -> str:
    trajectory_delta = _delta(one, two, "trajectory_confident_claim_fraction")
    median_delta = _delta(one, two, "median_family_margin_per_spike")
    imm_delta = _delta(one, two, "first_order_imm_raw_best_fraction")
    one_trajectory = _series_float(one, "trajectory_confident_claim_fraction")
    if np.isfinite(trajectory_delta) and trajectory_delta <= -abs(weaker_fraction_delta):
        return "weaker_1d_signal"
    if (
        np.isfinite(one_trajectory)
        and one_trajectory >= 0.5
        and np.isfinite(imm_delta)
        and imm_delta <= -abs(weaker_fraction_delta)
    ):
        return "strong_trajectory_family_but_weaker_imm_dominance"
    if np.isfinite(trajectory_delta) and abs(trajectory_delta) <= abs(similar_fraction_delta):
        if not np.isfinite(median_delta) or median_delta >= -abs(similar_fraction_delta):
            return "similarly_strong_1d_signal"
    return "mixed_1d_result"


def _statement_for_directional_pattern(pattern: str) -> str:
    if pattern == "weaker_1d_signal":
        return (
            "The scaled 1D comparison supports a weaker trajectory-family/IMM signal in constrained Z-track replay "
            "than in 2D open-field replay."
        )
    if pattern == "similarly_strong_1d_signal":
        return "The scaled 1D comparison supports a trajectory-family signal that generalizes beyond 2D open-field replay."
    if pattern == "strong_trajectory_family_but_weaker_imm_dominance":
        return (
            "The scaled 1D comparison supports trajectory-family replay in 1D, but with weaker first-order IMM dominance "
            "than in 2D."
        )
    return "The scaled 1D comparison is mixed and should be stratified by session, animal, detector settings, and track variables."


def _hard_caveat() -> str:
    return (
        "Do not claim IMM is only apparent in 2D without a robust weak or negative 1D result; "
        "single-animal/day smoke results remain hypothesis-generating."
    )


def _gate(gate: str, passed: bool, value: str, requirement: str, note: str) -> dict[str, object]:
    return {
        "gate": gate,
        "passed": bool(passed),
        "status": "pass" if passed else "fail",
        "value": value,
        "requirement": requirement,
        "note": note,
    }


def _one_d_summary_row(summary: pd.DataFrame) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=object)
    one_d = summary[summary["environment_type"].astype(str).str.startswith("1D")]
    if one_d.empty:
        return summary.iloc[0]
    return one_d.iloc[0]


def _animal_count(scores: pd.DataFrame) -> int:
    for column in ("animal", "rat", "source_animal"):
        if column in scores:
            values = scores[column].dropna().astype(str)
            values = values[values != ""]
            if not values.empty:
                return int(values.nunique())
    if "session" not in scores:
        return 0
    animals = scores["session"].dropna().astype(str).str.replace("\\", "/", regex=False).str.split("/").str[0]
    animals = animals[animals != ""]
    return int(animals.nunique())


def _linearization_diagnostic_values(diagnostics: str | Path | pd.DataFrame | None) -> dict[str, object]:
    if diagnostics is None:
        return _empty_linearization_values(False)
    frame = _load_optional_table(diagnostics)
    if frame is None or frame.empty:
        return _empty_linearization_values(False)
    metrics = _metric_value_map(frame)
    fraction = _metric_float(metrics, "fraction_valid_position", default=np.nan)
    median_projection = _metric_float(metrics, "median_projection_error_cm", default=0.0)
    track_length = _metric_float(metrics, "track_length_cm", default=np.nan)
    occupied = _occupied_linear_bins(frame, metrics)
    return {
        "available": True,
        "fraction_valid_position": fraction,
        "median_projection_error_cm": median_projection,
        "track_length_cm": track_length,
        "occupied_linear_bins": occupied,
    }


def _empty_linearization_values(available: bool) -> dict[str, object]:
    return {
        "available": bool(available),
        "fraction_valid_position": np.nan,
        "median_projection_error_cm": np.inf,
        "track_length_cm": np.nan,
        "occupied_linear_bins": 0,
    }


def _event_detection_values(summary: str | Path | pd.DataFrame | None, events: pd.DataFrame) -> dict[str, float]:
    frame = _load_optional_table(summary)
    if frame is not None and not frame.empty:
        event_candidates = _first_available_numeric(
            frame,
            ("ripple_events", "event_candidates", "candidate_events", "events", "event_count", "n_events"),
        )
        median_spikes = _first_available_numeric(
            frame,
            ("median_event_spikes", "median_spikes_per_event", "event_spikes_median", "median_n_spikes"),
        )
    else:
        event_candidates = np.nan
        median_spikes = np.nan
    if not np.isfinite(event_candidates):
        event_candidates = float(len(events))
    if not np.isfinite(median_spikes) and "n_spikes" in events:
        median_spikes = _numeric_median(events, "n_spikes")
    return {
        "event_candidates": float(event_candidates) if np.isfinite(event_candidates) else 0.0,
        "median_event_spikes": float(median_spikes) if np.isfinite(median_spikes) else 0.0,
    }


def _normalized_columns_present(one_d_summary: pd.Series) -> dict[str, object]:
    required = (
        "mean_family_margin_per_spike",
        "median_family_margin_per_spike",
        "mean_family_margin_per_time_bin",
        "median_family_margin_per_time_bin",
    )
    missing = [column for column in required if column not in one_d_summary.index]
    finite = []
    for column in required:
        value = _series_float(one_d_summary, column)
        finite.append(bool(np.isfinite(value)))
    passed = not missing and all(finite)
    return {
        "passed": passed,
        "value": "missing=" + ",".join(missing) + f"; finite={sum(finite)}/{len(required)}",
    }


def _load_optional_table(table: str | Path | pd.DataFrame | None) -> pd.DataFrame | None:
    if table is None:
        return None
    if isinstance(table, pd.DataFrame):
        return table.copy()
    path = Path(table)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _metric_value_map(frame: pd.DataFrame) -> dict[str, object]:
    if "metric" in frame and "value" in frame:
        return dict(zip(frame["metric"].astype(str), frame["value"], strict=False))
    if len(frame) == 1:
        return frame.iloc[0].to_dict()
    return {}


def _metric_float(metrics: dict[str, object], key: str, *, default: float) -> float:
    if key not in metrics:
        return float(default)
    value = pd.to_numeric(pd.Series([metrics[key]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float(default)


def _occupied_linear_bins(frame: pd.DataFrame, metrics: dict[str, object]) -> int:
    for key in ("occupied_linear_bins", "nonzero_occupancy_bins"):
        if key in metrics:
            value = _metric_float(metrics, key, default=0.0)
            return int(value) if np.isfinite(value) else 0
    if {"metric", "value"}.issubset(frame.columns):
        occupancy = frame[frame["metric"].astype(str).eq("occupancy_by_linear_bin")]
        values = pd.to_numeric(occupancy["value"], errors="coerce").fillna(0.0)
        return int((values > 0).sum())
    return 0


def _first_available_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> float:
    if len(frame) == 0:
        return np.nan
    for column in columns:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[0])
    return np.nan


def _exact_core_rows(scores: pd.DataFrame) -> pd.DataFrame:
    frame = scores.copy()
    if "status" not in frame:
        frame["status"] = "success"
    frame = ensure_evidence_support_columns(frame)
    if "evidence_comparable" in frame:
        frame["evidence_comparable"] = _coerce_bool_series(frame["evidence_comparable"])
    if "model" not in frame or "session" not in frame or "event_index" not in frame:
        return pd.DataFrame()
    status_ok = frame["status"].astype(str).str.lower().eq("success")
    comparable = _coerce_bool_series(frame["evidence_comparable"])
    exact = frame["evidence_support"].astype(str).eq(EXACT_EVIDENCE_SUPPORT)
    core = frame["model"].astype(str).isin(EXACT_CORE_MODELS)
    out = frame[status_ok & comparable & exact & core].copy()
    out["log_evidence"] = pd.to_numeric(out["log_evidence"], errors="coerce")
    return out.dropna(subset=["log_evidence"])


def _event_first_numeric(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return np.nan
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[0])


def _fallback_time_bins(group: pd.DataFrame) -> float:
    duration = _event_first_numeric(group, "duration_s")
    time_bin = _event_first_numeric(group, "time_bin_s")
    if np.isfinite(duration) and np.isfinite(time_bin) and duration > 0 and time_bin > 0:
        return float(duration / time_bin)
    return np.nan


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _mean_bool(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(frame[column].map(bool).mean())


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _numeric_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _series_float(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column, np.nan)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _delta(one: pd.Series, two: pd.Series, column: str) -> float:
    left = _series_float(one, column)
    right = _series_float(two, column)
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    return float(left - right)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olafsdottir-1d-evidence", required=True, type=Path, help="PR6 output directory or event-model-evidence CSV.")
    parser.add_argument("--pfeiffer-foster-2d-evidence", required=True, type=Path, help="2D full-core event-model-evidence CSV or artifact directory.")
    parser.add_argument("--output", default=Path("results/olafsdottir-1d-2d-comparison"), type=Path)
    parser.add_argument("--margin-threshold", default=5.5, type=float)
    parser.add_argument("--min-robust-1d-events", default=50, type=int)
    parser.add_argument("--min-1d-animals", default=2, type=int)
    parser.add_argument("--min-1d-sessions", default=2, type=int)
    parser.add_argument("--weaker-fraction-delta", default=0.20, type=float)
    parser.add_argument("--similar-fraction-delta", default=0.10, type=float)
    parser.add_argument("--one-d-dataset", default="Olafsdottir2016")
    parser.add_argument("--two-d-dataset", default="PfeifferFoster")
    parser.add_argument("--cell-identity-verified", action="store_true")
    parser.add_argument("--synthetic-1d-tests-passed", action="store_true")
    parser.add_argument("--linearization-diagnostics", type=Path, default=None)
    parser.add_argument("--event-detection-summary", type=Path, default=None)
    parser.add_argument("--min-linearization-valid-fraction", default=0.90, type=float)
    parser.add_argument("--max-linearization-median-projection-error-cm", default=15.0, type=float)
    parser.add_argument("--min-event-candidates", default=10, type=int)
    parser.add_argument("--min-event-median-spikes", default=5.0, type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    one_d = load_event_scores(args.olafsdottir_1d_evidence)
    two_d = load_event_scores(args.pfeiffer_foster_2d_evidence)
    tables = build_comparison(
        one_d_scores=one_d,
        two_d_scores=two_d,
        output=args.output,
        margin_threshold=args.margin_threshold,
        one_d_dataset=args.one_d_dataset,
        two_d_dataset=args.two_d_dataset,
        min_robust_1d_events=args.min_robust_1d_events,
        min_1d_animals=args.min_1d_animals,
        min_1d_sessions=args.min_1d_sessions,
        weaker_fraction_delta=args.weaker_fraction_delta,
        similar_fraction_delta=args.similar_fraction_delta,
        cell_identity_verified=args.cell_identity_verified,
        synthetic_1d_tests_passed=args.synthetic_1d_tests_passed,
        linearization_diagnostics=args.linearization_diagnostics,
        event_detection_summary=args.event_detection_summary,
        min_linearization_valid_fraction=args.min_linearization_valid_fraction,
        max_linearization_median_projection_error_cm=args.max_linearization_median_projection_error_cm,
        min_event_candidates=args.min_event_candidates,
        min_event_median_spikes=args.min_event_median_spikes,
    )
    print(tables["comparison_summary"].to_string(index=False))
    print()
    print(tables["interpretation_summary"].to_string(index=False))
    print()
    print(tables["readiness_gates"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
