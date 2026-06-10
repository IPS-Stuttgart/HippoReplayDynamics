#!/usr/bin/env python3
"""Triage synthetic momentum-recovery failures.

This script separates four cases that are otherwise easy to collapse into the
single, misleading statement "momentum recovery failed":

* strict exact-comparable recovery;
* lower-bound-certified recovery, where a truncated momentum lower bound already
  exceeds the best exact comparable model;
* candidate-support loss, where the synthetic true path is missing from the
  pruned support; and
* genuinely nondecisive lower bounds or exact non-recovery.

It is intended to be run on ``simulation_recovery_event_scores.csv`` or the
aggregate ``simulation_recovery_sweep_event_scores.csv``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_EXPECTED_MOMENTUM_MODEL = "sorted-spike-state-space-momentum"
EXACT_SUPPORT = "exact_full_grid"
TRUNCATED_SUPPORT = "truncated_full_grid"

SCORE_FILENAMES = (
    "simulation_recovery_sweep_event_scores.csv",
    "simulation_recovery_event_scores.csv",
    "event_scores.csv",
)

IDENTITY_COLUMN_CANDIDATES = (
    "matrix_id",
    "requested_session",
    "session",
    "simulation_event_index",
    "event_index",
    "replicate",
)

CONFIG_COLUMN_CANDIDATES = (
    "matrix_id",
    "requested_session",
    "state_space_diffusion_sigma_cm_sqrt_s",
    "state_space_momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay",
    "state_space_momentum_velocity_decay_tau_s",
    "state_space_momentum_candidate_top_k",
    "state_space_momentum_predicted_candidate_top_k",
    "state_space_momentum_candidate_source",
    "true_state_space_diffusion_sigma_cm_sqrt_s",
    "true_state_space_momentum_sigma_cm_sqrt_s",
    "true_state_space_momentum_initial_sigma_cm_sqrt_s",
    "true_state_space_momentum_velocity_decay",
    "true_state_space_momentum_velocity_decay_tau_s",
    "oracle_candidate_support",
)

SUPPORT_DIAGNOSTIC_COLUMNS = (
    "candidate_true_bin_coverage",
    "candidate_true_pair_coverage",
    "candidate_true_triplet_coverage",
    "candidate_true_path_fully_supported",
    "candidate_true_path_missing_bins",
)


@dataclass
class MomentumRecoveryTriageTables:
    """Output tables for momentum-recovery triage."""

    event_table: pd.DataFrame
    summary: pd.DataFrame
    failure_examples: pd.DataFrame

    def write(self, output: str | Path) -> None:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.event_table.to_csv(out_dir / "momentum_recovery_triage_event_table.csv", index=False)
        self.summary.to_csv(out_dir / "momentum_recovery_triage_summary.csv", index=False)
        self.failure_examples.to_csv(out_dir / "momentum_recovery_triage_failure_examples.csv", index=False)


def load_scores(path: str | Path) -> pd.DataFrame:
    """Load an event-score table from a CSV path or a result directory."""

    path = Path(path)
    if path.is_dir():
        for name in SCORE_FILENAMES:
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(
                f"{path} does not contain any of {', '.join(SCORE_FILENAMES)}"
            )
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return pd.read_csv(path)


def build_momentum_recovery_triage(
    scores: pd.DataFrame,
    *,
    expected_model: str = DEFAULT_EXPECTED_MOMENTUM_MODEL,
    max_examples: int = 50,
) -> MomentumRecoveryTriageTables:
    """Build event-level and config-level momentum-recovery diagnostics."""

    if scores.empty:
        raise ValueError("score table is empty")
    required = {"model", "log_evidence"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise KeyError(f"score table is missing required columns: {missing}")

    frame = _with_support_columns(scores)
    if "true_model" in frame.columns:
        frame = frame[frame["true_model"].astype(str).str.lower().eq("momentum")].copy()
    else:
        frame = frame[frame.get("expected_model", "").astype(str).eq(expected_model)].copy()
    if frame.empty:
        empty = pd.DataFrame()
        return MomentumRecoveryTriageTables(empty, empty, empty)

    rows: list[dict[str, object]] = []
    identity_columns = _present_columns(frame, IDENTITY_COLUMN_CANDIDATES)
    if not identity_columns:
        identity_columns = ["event_index"] if "event_index" in frame.columns else []

    groupby_keys: Sequence[str] | None = identity_columns or None
    groups = frame.groupby(list(groupby_keys), sort=False, dropna=False) if groupby_keys else [((), frame)]
    for key, group in groups:
        rows.append(_triage_event_group(group, expected_model=expected_model))

    event_table = pd.DataFrame(rows)
    if not event_table.empty:
        sort_columns = _present_columns(event_table, [*CONFIG_COLUMN_CANDIDATES, *IDENTITY_COLUMN_CANDIDATES])
        if sort_columns:
            event_table = event_table.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    summary = summarize_triage_events(event_table)
    failure_examples = _failure_examples(event_table, max_examples=max_examples)
    return MomentumRecoveryTriageTables(event_table, summary, failure_examples)


def summarize_triage_events(event_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize triage categories by configuration when possible."""

    if event_table.empty:
        return pd.DataFrame()
    group_columns = _present_columns(event_table, CONFIG_COLUMN_CANDIDATES)
    if not group_columns:
        group_columns = ["_all"]
        event_table = event_table.copy()
        event_table["_all"] = "all"

    rows: list[dict[str, object]] = []
    for _, group in event_table.groupby(group_columns, sort=False, dropna=False):
        row = {col: group.iloc[0][col] for col in group_columns if col != "_all"}
        row.update(
            {
                "momentum_events": int(len(group)),
                "strict_exact_recovery_events": int(
                    _bool_series(group["strict_exact_recovery"]).sum()
                ),
                "lower_bound_certified_recovery_events": int(
                    _bool_series(group["lower_bound_certified_recovery"]).sum()
                ),
                "certified_or_strict_recovery_events": int(
                    _bool_series(group["certified_or_strict_recovery"]).sum()
                ),
                "candidate_support_loss_events": int(
                    _bool_series(group["candidate_support_loss"]).sum()
                ),
                "nondecisive_lower_bound_events": int(
                    group["triage_category"].eq("nondecisive_lower_bound").sum()
                ),
                "exact_nonrecovery_events": int(
                    group["triage_category"].eq("exact_nonrecovery").sum()
                ),
                "mean_expected_minus_best_comparable_log_evidence": _mean(
                    group["expected_minus_best_comparable_log_evidence"]
                ),
                "median_expected_minus_best_comparable_log_evidence": _median(
                    group["expected_minus_best_comparable_log_evidence"]
                ),
                "mean_candidate_true_bin_coverage": _mean(
                    group.get("candidate_true_bin_coverage", pd.Series(dtype=float))
                ),
                "mean_candidate_true_triplet_coverage": _mean(
                    group.get("candidate_true_triplet_coverage", pd.Series(dtype=float))
                ),
                "mean_candidate_true_path_missing_bins": _mean(
                    group.get("candidate_true_path_missing_bins", pd.Series(dtype=float))
                ),
            }
        )
        denom = max(1, int(row["momentum_events"]))
        row["certified_or_strict_recovery_fraction"] = (
            row["certified_or_strict_recovery_events"] / denom
        )
        row["candidate_support_loss_fraction"] = row["candidate_support_loss_events"] / denom
        category_counts = group["triage_category"].value_counts()
        for category, count in category_counts.items():
            row[f"category_{category}_events"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows)


def _triage_event_group(group: pd.DataFrame, *, expected_model: str) -> dict[str, object]:
    first = group.iloc[0]
    row: dict[str, object] = {}
    for column in _present_columns(group, [*CONFIG_COLUMN_CANDIDATES, *IDENTITY_COLUMN_CANDIDATES]):
        row[column] = first[column]

    row["true_model"] = str(first.get("true_model", "momentum"))
    row["expected_model"] = str(first.get("expected_model", expected_model)) or expected_model
    row["expected_model_requested"] = expected_model
    row["scored_models"] = ";".join(sorted({str(model) for model in group["model"].dropna()}))

    scored = group[_success_mask(group)].copy()
    if scored.empty:
        return {
            **row,
            **_empty_decision("no_successful_scores"),
        }

    comparable = scored[_comparable_mask(scored)].copy()
    best_comparable_model = ""
    best_comparable_log_evidence = float("nan")
    if not comparable.empty:
        best_comparable = _best_log_evidence_row(comparable)
        best_comparable_model = str(best_comparable["model"])
        best_comparable_log_evidence = float(best_comparable["log_evidence"])

    expected_rows = scored[scored["model"].astype(str).eq(expected_model)].copy()
    if expected_rows.empty:
        return {
            **row,
            **_empty_decision("expected_model_not_scored"),
            "best_comparable_model": best_comparable_model,
            "best_comparable_log_evidence": best_comparable_log_evidence,
        }

    expected = _best_log_evidence_row(expected_rows)
    expected_log_evidence = float(expected["log_evidence"])
    expected_support = str(expected.get("evidence_support", EXACT_SUPPORT))
    expected_comparable = _as_bool(
        expected.get("evidence_comparable", expected_support == EXACT_SUPPORT)
    )
    margin = expected_log_evidence - best_comparable_log_evidence
    strict_exact_recovery = bool(expected_comparable and best_comparable_model == expected_model)
    lower_bound_certified_recovery = bool(
        (not expected_comparable)
        and expected_support == TRUNCATED_SUPPORT
        and math.isfinite(best_comparable_log_evidence)
        and margin > 0.0
    )
    certified_or_strict = strict_exact_recovery or lower_bound_certified_recovery

    support_values = _support_diagnostics(expected)
    support_loss = _candidate_support_loss(support_values)
    oracle_support = _as_bool(expected.get("oracle_candidate_support", first.get("oracle_candidate_support", False)))

    if oracle_support and certified_or_strict:
        category = "oracle_support_recovers"
    elif oracle_support and not certified_or_strict:
        category = "oracle_support_does_not_recover"
    elif strict_exact_recovery:
        category = "strict_exact_recovery"
    elif lower_bound_certified_recovery:
        category = "lower_bound_certified_recovery"
    elif expected_comparable:
        category = "exact_nonrecovery"
    elif not math.isfinite(best_comparable_log_evidence):
        category = "no_comparable_exact_reference"
    elif support_loss:
        category = "candidate_support_loss"
    else:
        category = "nondecisive_lower_bound"

    return {
        **row,
        "triage_category": category,
        "strict_exact_recovery": strict_exact_recovery,
        "lower_bound_certified_recovery": lower_bound_certified_recovery,
        "certified_or_strict_recovery": certified_or_strict,
        "candidate_support_loss": support_loss,
        "oracle_candidate_support": oracle_support,
        "expected_model_log_evidence": expected_log_evidence,
        "expected_model_evidence_support": expected_support,
        "expected_model_evidence_comparable": expected_comparable,
        "best_comparable_model": best_comparable_model,
        "best_comparable_log_evidence": best_comparable_log_evidence,
        "expected_minus_best_comparable_log_evidence": float(margin),
        **support_values,
    }


def _empty_decision(category: str) -> dict[str, object]:
    return {
        "triage_category": category,
        "strict_exact_recovery": False,
        "lower_bound_certified_recovery": False,
        "certified_or_strict_recovery": False,
        "candidate_support_loss": False,
        "oracle_candidate_support": False,
        "expected_model_log_evidence": float("nan"),
        "expected_model_evidence_support": "",
        "expected_model_evidence_comparable": False,
        "best_comparable_model": "",
        "best_comparable_log_evidence": float("nan"),
        "expected_minus_best_comparable_log_evidence": float("nan"),
    }


def _with_support_columns(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    if "evidence_support" not in out.columns:
        support_columns = [
            col
            for col in (
                "diagnostic_candidate_evidence_support",
                "diagnostic_state_space_momentum_evidence_support",
                "diagnostic_state_space_imm_evidence_support",
            )
            if col in out.columns
        ]
        if support_columns:
            out["evidence_support"] = out[support_columns].bfill(axis=1).iloc[:, 0].fillna(EXACT_SUPPORT)
        else:
            out["evidence_support"] = EXACT_SUPPORT
    if "evidence_comparable" not in out.columns:
        status_ok = out["status"].astype(str).eq("success") if "status" in out.columns else True
        out["evidence_comparable"] = status_ok & out["evidence_support"].astype(str).eq(EXACT_SUPPORT)
    return out


def _success_mask(frame: pd.DataFrame) -> pd.Series:
    status_ok = frame["status"].astype(str).eq("success") if "status" in frame.columns else pd.Series(True, index=frame.index)
    values = pd.to_numeric(frame["log_evidence"], errors="coerce")
    return status_ok & np.isfinite(values)


def _comparable_mask(frame: pd.DataFrame) -> pd.Series:
    if "evidence_comparable" in frame.columns:
        return _bool_series(frame["evidence_comparable"])
    return frame["evidence_support"].astype(str).eq(EXACT_SUPPORT)


def _best_log_evidence_row(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame["log_evidence"], errors="coerce").to_numpy(float)
    return frame.iloc[int(np.nanargmax(values))]


def _support_diagnostics(row: pd.Series) -> dict[str, object]:
    values: dict[str, object] = {}
    for column in SUPPORT_DIAGNOSTIC_COLUMNS:
        values[column] = row[column] if column in row.index else float("nan")
    return values


def _candidate_support_loss(values: dict[str, object]) -> bool:
    missing = _numeric(values.get("candidate_true_path_missing_bins"))
    if math.isfinite(missing) and missing > 0:
        return True
    fully_supported = _numeric(values.get("candidate_true_path_fully_supported"))
    if math.isfinite(fully_supported) and fully_supported < 1:
        return True
    for column in ("candidate_true_bin_coverage", "candidate_true_pair_coverage", "candidate_true_triplet_coverage"):
        coverage = _numeric(values.get(column))
        if math.isfinite(coverage) and coverage < 1.0:
            return True
    return False


def _failure_examples(event_table: pd.DataFrame, *, max_examples: int) -> pd.DataFrame:
    if event_table.empty:
        return pd.DataFrame()
    failures = event_table[
        ~_bool_series(event_table["certified_or_strict_recovery"])
    ].copy()
    if failures.empty:
        return pd.DataFrame()
    priority = {
        "candidate_support_loss": 0,
        "oracle_support_does_not_recover": 1,
        "exact_nonrecovery": 2,
        "nondecisive_lower_bound": 3,
        "no_comparable_exact_reference": 4,
        "expected_model_not_scored": 5,
        "no_successful_scores": 6,
    }
    failures["_priority"] = failures["triage_category"].map(priority).fillna(99).astype(int)
    failures["_missing"] = pd.to_numeric(
        failures.get("candidate_true_path_missing_bins", pd.Series(0, index=failures.index)),
        errors="coerce",
    ).fillna(0)
    failures["_margin"] = pd.to_numeric(
        failures.get("expected_minus_best_comparable_log_evidence", pd.Series(float("nan"), index=failures.index)),
        errors="coerce",
    ).fillna(float("-inf"))
    out = failures.sort_values(["_priority", "_missing", "_margin"], ascending=[True, False, True], kind="stable")
    return out.drop(columns=["_priority", "_missing", "_margin"]).head(max_examples).reset_index(drop=True)


def _present_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def _numeric(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(math.isfinite(numeric) and numeric != 0.0)
    return str(value).strip().lower() in {"1", "1.0", "true", "t", "yes", "y", "on"}


def _bool_series(values: pd.Series) -> pd.Series:
    return values.map(_as_bool).astype(bool)


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float("nan") if numeric.empty else float(numeric.mean())


def _median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float("nan") if numeric.empty else float(numeric.median())


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage synthetic momentum-recovery failures.")
    parser.add_argument("--scores", required=True, help="Simulation-recovery event-score CSV or result directory.")
    parser.add_argument("--output", required=True, help="Output directory for triage CSVs.")
    parser.add_argument("--expected-model", default=DEFAULT_EXPECTED_MOMENTUM_MODEL)
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    tables = build_momentum_recovery_triage(
        load_scores(args.scores),
        expected_model=args.expected_model,
        max_examples=args.max_examples,
    )
    tables.write(args.output)
    if tables.summary.empty:
        print("No true momentum events were available for triage.")
    else:
        print(tables.summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
