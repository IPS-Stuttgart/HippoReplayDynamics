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
    weaker_fraction_delta: float = 0.20,
    similar_fraction_delta: float = 0.10,
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
    interpretation = interpretation_summary(
        summary,
        margin_threshold=margin_threshold,
        min_robust_1d_events=min_robust_1d_events,
        weaker_fraction_delta=weaker_fraction_delta,
        similar_fraction_delta=similar_fraction_delta,
    )
    summary.to_csv(output_dir / SUMMARY_OUTPUT, index=False)
    interpretation.to_csv(output_dir / INTERPRETATION_OUTPUT, index=False)
    return {
        "comparison_summary": summary,
        "interpretation_summary": interpretation,
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
    if data_limited:
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
        "margin_threshold": float(margin_threshold),
        "min_robust_1d_events": int(min_robust_1d_events),
    }
    return pd.DataFrame([row], columns=columns)


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
    parser.add_argument("--weaker-fraction-delta", default=0.20, type=float)
    parser.add_argument("--similar-fraction-delta", default=0.10, type=float)
    parser.add_argument("--one-d-dataset", default="Olafsdottir2016")
    parser.add_argument("--two-d-dataset", default="PfeifferFoster")
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
        weaker_fraction_delta=args.weaker_fraction_delta,
        similar_fraction_delta=args.similar_fraction_delta,
    )
    print(tables["comparison_summary"].to_string(index=False))
    print()
    print(tables["interpretation_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
