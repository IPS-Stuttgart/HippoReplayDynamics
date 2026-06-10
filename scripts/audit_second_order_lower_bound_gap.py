#!/usr/bin/env python3
"""Audit candidate-pruned second-order lower-bound gaps.

Compare truncated momentum/IMM evidence rows with exact full-grid rows for the
same event, model, and dynamics parameters when both rows are present.  This is
an audit/reporting utility; it does not generate exact rows itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


EXACT_SUPPORT = "exact_full_grid"
TRUNCATED_SUPPORT = "truncated_full_grid"
DEFAULT_MODELS = (
    "sorted-spike-state-space-momentum",
    "sorted-spike-state-space-imm",
    "clusterless-state-space-momentum",
    "clusterless-state-space-imm",
)
SCORE_FILENAMES = (
    "event_scores.csv",
    "model_evidence_event_scores.csv",
    "simulation_recovery_event_scores.csv",
    "simulation_recovery_sweep_event_scores.csv",
)
VALUE_COLUMN_CANDIDATES = ("log_evidence", "heldout_log_likelihood", "log_likelihood")
EVENT_ID_COLUMNS = (
    "session",
    "requested_session",
    "event_index",
    "simulation_event_index",
    "benchmark_cell_split_index",
    "benchmark_cell_split_seed",
    "random_seed",
)
MATCH_PARAMETER_COLUMNS = (
    "time_bin_ms",
    "time_bin_s",
    "state_space_stationary_sigma_cm",
    "state_space_diffusion_sigma_cm_sqrt_s",
    "state_space_max_step_sigma",
    "state_space_imm_mode_stickiness",
    "state_space_momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay",
    "state_space_momentum_velocity_decay_tau_s",
    "state_space_valid_occupancy_threshold_s",
    "scoring_state_space_stationary_sigma_cm",
    "scoring_state_space_diffusion_sigma_cm_sqrt_s",
    "scoring_state_space_max_step_sigma",
    "scoring_state_space_imm_mode_stickiness",
    "scoring_state_space_momentum_sigma_cm_sqrt_s",
    "scoring_state_space_momentum_initial_sigma_cm_sqrt_s",
    "scoring_state_space_momentum_velocity_decay",
    "scoring_state_space_momentum_velocity_decay_tau_s",
)
TRUNCATED_DIAGNOSTIC_COLUMNS = (
    "matrix_id",
    "state_space_momentum_candidate_top_k",
    "state_space_momentum_predicted_candidate_top_k",
    "state_space_momentum_candidate_source",
    "mean_candidate_log_mass",
    "min_candidate_log_mass",
    "mean_candidate_count",
    "candidate_true_bin_coverage",
    "candidate_true_pair_coverage",
    "candidate_true_triplet_coverage",
    "candidate_true_path_missing_bins",
)


@dataclass
class LowerBoundGapTables:
    event_gaps: pd.DataFrame
    summary: pd.DataFrame

    def write(self, output: str | Path) -> None:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.event_gaps.to_csv(out_dir / "second_order_lower_bound_gap_event_table.csv", index=False)
        self.summary.to_csv(out_dir / "second_order_lower_bound_gap_summary.csv", index=False)


def load_score_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        for name in SCORE_FILENAMES:
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"{path} does not contain a known score CSV")
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return pd.read_csv(path)


def build_lower_bound_gap_tables(
    scores: pd.DataFrame,
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    value_column: str | None = None,
) -> LowerBoundGapTables:
    if scores.empty:
        return LowerBoundGapTables(pd.DataFrame(), pd.DataFrame())
    frame = _with_support_columns(scores)
    value_column = _resolve_value_column(frame, value_column)
    required = {"model", "evidence_support", value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"score table is missing required columns: {missing}")

    model_set = {str(model) for model in models}
    frame = frame[frame["model"].astype(str).isin(model_set)].copy()
    if frame.empty:
        return LowerBoundGapTables(pd.DataFrame(), pd.DataFrame())
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame[np.isfinite(frame[value_column])].copy()
    if frame.empty:
        return LowerBoundGapTables(pd.DataFrame(), pd.DataFrame())

    match_columns = _present_columns(frame, [*EVENT_ID_COLUMNS, "model", *MATCH_PARAMETER_COLUMNS])
    if "model" not in match_columns:
        match_columns.append("model")
    exact = frame[frame["evidence_support"].astype(str).eq(EXACT_SUPPORT)].copy()
    truncated = frame[frame["evidence_support"].astype(str).eq(TRUNCATED_SUPPORT)].copy()
    if exact.empty or truncated.empty:
        return LowerBoundGapTables(pd.DataFrame(), pd.DataFrame())

    exact_summary = (
        exact.groupby(match_columns, sort=False, dropna=False)
        .agg(exact_log_evidence=(value_column, "max"), exact_source_rows=(value_column, "size"))
        .reset_index()
    )
    keep_columns = [*match_columns, value_column, *_present_columns(truncated, TRUNCATED_DIAGNOSTIC_COLUMNS)]
    truncated = truncated[keep_columns].rename(columns={value_column: "truncated_lower_bound_log_evidence"})
    event_gaps = truncated.merge(exact_summary, on=match_columns, how="inner")
    if event_gaps.empty:
        return LowerBoundGapTables(event_gaps, pd.DataFrame())

    event_gaps["lower_bound_gap_log_evidence"] = (
        event_gaps["exact_log_evidence"] - event_gaps["truncated_lower_bound_log_evidence"]
    )
    event_gaps["lower_bound_exceeds_exact"] = event_gaps["lower_bound_gap_log_evidence"] < -1e-6
    event_gaps["lower_bound_gap_within_1"] = event_gaps["lower_bound_gap_log_evidence"].abs() <= 1.0
    event_gaps["lower_bound_gap_within_3"] = event_gaps["lower_bound_gap_log_evidence"].abs() <= 3.0
    event_gaps["lower_bound_gap_within_10"] = event_gaps["lower_bound_gap_log_evidence"].abs() <= 10.0
    event_gaps = event_gaps.sort_values(match_columns, kind="stable").reset_index(drop=True)
    return LowerBoundGapTables(event_gaps, summarize_gap_table(event_gaps))


def summarize_gap_table(event_gaps: pd.DataFrame) -> pd.DataFrame:
    if event_gaps.empty:
        return pd.DataFrame()
    group_columns = _present_columns(
        event_gaps,
        [
            "model",
            "state_space_momentum_candidate_top_k",
            "state_space_momentum_predicted_candidate_top_k",
            "state_space_momentum_candidate_source",
        ],
    )
    if not group_columns:
        group_columns = ["_all"]
        event_gaps = event_gaps.copy()
        event_gaps["_all"] = "all"

    rows: list[dict[str, object]] = []
    for _, group in event_gaps.groupby(group_columns, sort=False, dropna=False):
        gaps = pd.to_numeric(group["lower_bound_gap_log_evidence"], errors="coerce").dropna()
        row = {column: group.iloc[0][column] for column in group_columns if column != "_all"}
        row.update(
            {
                "paired_event_rows": int(len(group)),
                "events_with_negative_gap": int(_bool_series(group, "lower_bound_exceeds_exact").sum()),
                "mean_lower_bound_gap_log_evidence": _safe_stat(gaps, "mean"),
                "median_lower_bound_gap_log_evidence": _safe_stat(gaps, "median"),
                "p95_lower_bound_gap_log_evidence": _safe_quantile(gaps, 0.95),
                "max_lower_bound_gap_log_evidence": _safe_stat(gaps, "max"),
                "gap_within_1_fraction": _bool_fraction(group, "lower_bound_gap_within_1"),
                "gap_within_3_fraction": _bool_fraction(group, "lower_bound_gap_within_3"),
                "gap_within_10_fraction": _bool_fraction(group, "lower_bound_gap_within_10"),
                "mean_min_candidate_log_mass": _safe_stat(
                    pd.to_numeric(group.get("min_candidate_log_mass", pd.Series(dtype=float)), errors="coerce").dropna(),
                    "mean",
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _with_support_columns(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    if "evidence_support" not in out.columns:
        support_columns = [
            column
            for column in (
                "diagnostic_candidate_evidence_support",
                "diagnostic_state_space_momentum_evidence_support",
                "diagnostic_state_space_imm_evidence_support",
            )
            if column in out.columns
        ]
        out["evidence_support"] = (
            out[support_columns].bfill(axis=1).iloc[:, 0].fillna(EXACT_SUPPORT)
            if support_columns
            else EXACT_SUPPORT
        )
    return out


def _as_bool(value: object) -> bool:
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
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y"}


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool).astype(bool)


def _bool_fraction(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(_bool_series(frame, column).mean())


def _resolve_value_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise KeyError(f"requested value column {requested!r} is absent")
        return requested
    for column in VALUE_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column
    raise KeyError(f"none of the value columns are present: {VALUE_COLUMN_CANDIDATES}")


def _present_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _safe_stat(values: pd.Series, name: str) -> float:
    if values.empty:
        return float("nan")
    if name == "mean":
        return float(values.mean())
    if name == "median":
        return float(values.median())
    if name == "max":
        return float(values.max())
    raise ValueError(f"unknown stat: {name}")


def _safe_quantile(values: pd.Series, q: float) -> float:
    return float("nan") if values.empty else float(values.quantile(q))


def _parse_models(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.replace(",", " ").split() if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit candidate-pruned second-order lower-bound gaps.")
    parser.add_argument("--scores", required=True, help="Score CSV or directory containing a score CSV.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--models", default=" ".join(DEFAULT_MODELS))
    parser.add_argument("--value-column", default=None)
    args = parser.parse_args()

    tables = build_lower_bound_gap_tables(
        load_score_table(args.scores),
        models=_parse_models(args.models),
        value_column=args.value_column,
    )
    tables.write(args.output)
    if tables.summary.empty:
        print("No exact/truncated second-order pairs were available for gap audit.")
    else:
        print(tables.summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
