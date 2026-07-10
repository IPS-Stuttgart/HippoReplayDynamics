#!/usr/bin/env python3
"""Select a synthetic-calibrated momentum confidence threshold."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    infer_paired_model_group_cols,
    paired_model_margin_decisions,
    paired_model_margin_summary,
    paired_model_margin_threshold_sweep,
    select_paired_model_margin_threshold,
)


DEFAULT_POSITIVE_MODEL = "sorted-spike-state-space-momentum-exact-sparse"
DEFAULT_REFERENCE_MODEL = "sorted-spike-state-space-diffusion"
DEFAULT_THRESHOLDS = tuple(value / 2.0 for value in range(0, 21))
SCORE_FILE_CANDIDATES = (
    "simulation_recovery_emission_calibration_event_scores.csv",
    "simulation_recovery_sweep_event_scores.csv",
    "simulation_recovery_event_scores.csv",
    "state_space_evidence_sweep_event_scores.csv",
    "all_sessions_event_model_evidence.csv",
    "event_model_evidence.csv",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-scores", required=True, help="Synthetic recovery event-score CSV or artifact directory.")
    parser.add_argument("--evidence-scores", help="Optional real replay event-score CSV or artifact directory.")
    parser.add_argument("--output", default="results/momentum-confidence-threshold")
    parser.add_argument("--positive-model", default=DEFAULT_POSITIVE_MODEL)
    parser.add_argument("--reference-model", default=DEFAULT_REFERENCE_MODEL)
    parser.add_argument("--thresholds", default=" ".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--recovery-group-cols", default="", help="Comma-separated paired-event grouping columns.")
    parser.add_argument("--evidence-group-cols", default="", help="Comma-separated paired-event grouping columns.")
    parser.add_argument("--true-model-column", default="true_model")
    parser.add_argument("--positive-true-label", default="momentum")
    parser.add_argument("--stratify-column", default="matrix_id")
    parser.add_argument(
        "--stratify-columns",
        default="",
        help=(
            "Comma-separated parameter columns for stratum-specific summaries. "
            "When omitted, --stratify-column is used."
        ),
    )
    parser.add_argument(
        "--threshold-scope",
        choices=("global", "stratum"),
        default="global",
        help="Select one global threshold or one threshold per stratification cell.",
    )
    parser.add_argument("--max-false-positive-claims", type=int, default=0)
    parser.add_argument("--min-positive-claim-recall", type=float, default=0.0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_float_values(args.thresholds)
    stratify_cols = _parse_stratify_cols(args.stratify_columns, args.stratify_column)
    recovery_scores_path = find_score_file(args.recovery_scores)
    recovery = _ensure_stratify_columns(pd.read_csv(recovery_scores_path), stratify_cols)
    recovery_group_cols = _parse_group_cols(args.recovery_group_cols) or infer_paired_model_group_cols(recovery)
    if args.threshold_scope == "stratum":
        recovery_sweep = _threshold_sweep_by_strata(
            recovery,
            stratify_cols=stratify_cols,
            positive_model=args.positive_model,
            reference_model=args.reference_model,
            thresholds=thresholds,
            group_cols=recovery_group_cols,
            true_model_col=args.true_model_column,
            positive_true_label=args.positive_true_label,
        )
        selection = _select_thresholds_by_strata(
            recovery_sweep,
            stratify_cols=stratify_cols,
            max_false_positive_claims=args.max_false_positive_claims,
            min_positive_claim_recall=args.min_positive_claim_recall,
        )
        recovery_by_stratum = _summaries_by_strata_with_thresholds(
            recovery,
            stratify_cols=stratify_cols,
            thresholds=selection,
            positive_model=args.positive_model,
            reference_model=args.reference_model,
            true_model_col=args.true_model_column,
            positive_true_label=args.positive_true_label,
        )
    else:
        recovery_sweep = paired_model_margin_threshold_sweep(
            recovery,
            positive_model=args.positive_model,
            reference_model=args.reference_model,
            thresholds=thresholds,
            group_cols=recovery_group_cols,
            true_model_col=args.true_model_column,
            positive_true_label=args.positive_true_label,
        )
        selection = select_paired_model_margin_threshold(
            recovery_sweep,
            max_false_positive_claims=args.max_false_positive_claims,
            min_positive_claim_recall=args.min_positive_claim_recall,
        )
        selected_threshold = float(selection["selected_margin_threshold"].iloc[0])
        recovery_by_stratum = _summaries_by_strata(
            recovery,
            stratify_cols=stratify_cols,
            threshold=selected_threshold,
            positive_model=args.positive_model,
            reference_model=args.reference_model,
            true_model_col=args.true_model_column,
            positive_true_label=args.positive_true_label,
        )
    recovery_sweep["threshold_scope"] = args.threshold_scope
    selection["threshold_scope"] = args.threshold_scope
    recovery_by_stratum["threshold_scope"] = args.threshold_scope
    recovery_sweep.to_csv(out / "momentum_confidence_threshold_recovery_sweep.csv", index=False)
    selection.to_csv(out / "momentum_confidence_threshold_selection.csv", index=False)
    recovery_by_stratum.to_csv(out / "momentum_confidence_threshold_recovery_by_stratum.csv", index=False)

    print(selection.to_string(index=False))

    if args.evidence_scores:
        evidence_scores_path = find_score_file(args.evidence_scores)
        evidence = _ensure_stratify_columns(pd.read_csv(evidence_scores_path), stratify_cols)
        evidence_group_cols = _parse_group_cols(args.evidence_group_cols) or infer_paired_model_group_cols(evidence)
        if args.threshold_scope == "stratum":
            evidence_sweep = _threshold_sweep_by_strata(
                evidence,
                stratify_cols=stratify_cols,
                positive_model=args.positive_model,
                reference_model=args.reference_model,
                thresholds=thresholds,
                group_cols=evidence_group_cols,
            )
            evidence_decisions = _decisions_by_strata_with_thresholds(
                evidence,
                stratify_cols=stratify_cols,
                thresholds=selection,
                positive_model=args.positive_model,
                reference_model=args.reference_model,
            )
            evidence_by_stratum = _summaries_by_strata_with_thresholds(
                evidence,
                stratify_cols=stratify_cols,
                thresholds=selection,
                positive_model=args.positive_model,
                reference_model=args.reference_model,
            )
            selected_threshold = float("nan")
        else:
            evidence_sweep = paired_model_margin_threshold_sweep(
                evidence,
                positive_model=args.positive_model,
                reference_model=args.reference_model,
                thresholds=thresholds,
                group_cols=evidence_group_cols,
            )
            evidence_decisions = paired_model_margin_decisions(
                evidence,
                positive_model=args.positive_model,
                reference_model=args.reference_model,
                margin_threshold=selected_threshold,
                group_cols=evidence_group_cols,
            )
            evidence_by_stratum = _summaries_by_strata(
                evidence,
                stratify_cols=stratify_cols,
                threshold=selected_threshold,
                positive_model=args.positive_model,
                reference_model=args.reference_model,
            )
        evidence_summary = paired_model_margin_summary(evidence_decisions)
        evidence_summary["selected_margin_threshold"] = selected_threshold
        evidence_summary["threshold_scope"] = args.threshold_scope
        evidence_summary["recovery_scores"] = str(recovery_scores_path)
        evidence_summary["evidence_scores"] = str(evidence_scores_path)
        evidence_summary["recovery_group_cols"] = ",".join(recovery_group_cols)
        evidence_summary["evidence_group_cols"] = ",".join(evidence_group_cols)
        evidence_sweep["threshold_scope"] = args.threshold_scope
        evidence_decisions["threshold_scope"] = args.threshold_scope
        evidence_by_stratum["threshold_scope"] = args.threshold_scope
        evidence_sweep.to_csv(out / "momentum_confidence_threshold_evidence_sweep.csv", index=False)
        evidence_decisions.to_csv(out / "momentum_confidence_threshold_evidence_decisions.csv", index=False)
        evidence_summary.to_csv(out / "momentum_confidence_threshold_evidence_summary.csv", index=False)
        evidence_by_stratum.to_csv(out / "momentum_confidence_threshold_evidence_by_stratum.csv", index=False)
        print("\nEvidence at selected threshold:")
        print(evidence_summary.to_string(index=False))

    return 0


def find_score_file(root: str | Path) -> Path:
    path = Path(root)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    matches: list[Path] = []
    for candidate in SCORE_FILE_CANDIDATES:
        matches.extend(sorted(path.rglob(candidate)))
    if not matches:
        raise FileNotFoundError(f"no supported score file found under {path}")
    return matches[0]


def _parse_float_values(value: str) -> tuple[float, ...]:
    normalized = value.replace(",", " ")
    values = tuple(float(item) for item in normalized.split() if item)
    if not values:
        raise ValueError("at least one threshold is required")
    return values


def _parse_group_cols(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_stratify_cols(columns: str, column: str) -> tuple[str, ...]:
    parsed = _parse_group_cols(columns)
    if parsed:
        return parsed
    parsed_column = _parse_group_cols(column)
    return parsed_column or ("matrix_id",)


def _ensure_stratify_columns(scores: pd.DataFrame, stratify_cols: Sequence[str]) -> pd.DataFrame:
    out = scores.copy()
    if "time_bin_s" in stratify_cols and "time_bin_s" not in out and "time_bin_ms" in out:
        out["time_bin_s"] = pd.to_numeric(out["time_bin_ms"], errors="raise") / 1000.0
    missing = [column for column in stratify_cols if column not in out.columns]
    if missing:
        raise KeyError(f"scores are missing stratification columns: {missing}")
    return out


def _threshold_sweep_by_strata(
    scores: pd.DataFrame,
    *,
    stratify_cols: Sequence[str],
    positive_model: str,
    reference_model: str,
    thresholds: Sequence[float],
    group_cols: Sequence[str],
    true_model_col: str | None = None,
    positive_true_label: str | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key, group in _iter_strata(scores, stratify_cols):
        sweep = paired_model_margin_threshold_sweep(
            group,
            positive_model=positive_model,
            reference_model=reference_model,
            thresholds=thresholds,
            group_cols=group_cols,
            true_model_col=true_model_col if true_model_col in group.columns else None,
            positive_true_label=positive_true_label,
        )
        rows.append(_prepend_stratum_values(sweep, stratify_cols, key))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _select_thresholds_by_strata(
    threshold_sweep: pd.DataFrame,
    *,
    stratify_cols: Sequence[str],
    max_false_positive_claims: int,
    min_positive_claim_recall: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key, group in _iter_strata(threshold_sweep, stratify_cols):
        selected = select_paired_model_margin_threshold(
            group,
            max_false_positive_claims=max_false_positive_claims,
            min_positive_claim_recall=min_positive_claim_recall,
        )
        rows.append(_prepend_stratum_values(selected, stratify_cols, key))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _summaries_by_strata(
    scores: pd.DataFrame,
    *,
    stratify_cols: Sequence[str],
    threshold: float,
    positive_model: str,
    reference_model: str,
    true_model_col: str | None = None,
    positive_true_label: str | None = None,
) -> pd.DataFrame:
    thresholds = _constant_thresholds_for_strata(scores, stratify_cols, threshold)
    return _summaries_by_strata_with_thresholds(
        scores,
        stratify_cols=stratify_cols,
        thresholds=thresholds,
        positive_model=positive_model,
        reference_model=reference_model,
        true_model_col=true_model_col,
        positive_true_label=positive_true_label,
    )


def _summaries_by_strata_with_thresholds(
    scores: pd.DataFrame,
    *,
    stratify_cols: Sequence[str],
    thresholds: pd.DataFrame,
    positive_model: str,
    reference_model: str,
    true_model_col: str | None = None,
    positive_true_label: str | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key, group in _iter_strata(scores, stratify_cols):
        threshold = _threshold_for_key(thresholds, stratify_cols, key)
        decisions = paired_model_margin_decisions(
            group,
            positive_model=positive_model,
            reference_model=reference_model,
            margin_threshold=threshold,
            group_cols=infer_paired_model_group_cols(group),
            true_model_col=true_model_col if true_model_col in group.columns else None,
            positive_true_label=positive_true_label,
        )
        summary = paired_model_margin_summary(
            decisions,
            true_model_col=true_model_col if true_model_col in group.columns else None,
        )
        summary = _prepend_stratum_values(summary, stratify_cols, key)
        summary = _append_unique_context_columns(summary, group, stratify_cols)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _decisions_by_strata_with_thresholds(
    scores: pd.DataFrame,
    *,
    stratify_cols: Sequence[str],
    thresholds: pd.DataFrame,
    positive_model: str,
    reference_model: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for key, group in _iter_strata(scores, stratify_cols):
        threshold = _threshold_for_key(thresholds, stratify_cols, key)
        rows.append(
            paired_model_margin_decisions(
                group,
                positive_model=positive_model,
                reference_model=reference_model,
                margin_threshold=threshold,
                group_cols=infer_paired_model_group_cols(group),
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _constant_thresholds_for_strata(
    scores: pd.DataFrame, stratify_cols: Sequence[str], threshold: float
) -> pd.DataFrame:
    rows = []
    for key, _ in _iter_strata(scores, stratify_cols):
        row = dict(zip(stratify_cols, key, strict=True))
        row["selected_margin_threshold"] = float(threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_for_key(thresholds: pd.DataFrame, stratify_cols: Sequence[str], key: tuple[object, ...]) -> float:
    matches = thresholds
    for column, value in zip(stratify_cols, key, strict=True):
        if pd.isna(value):
            matches = matches[matches[column].isna()]
        else:
            matches = matches[matches[column].eq(value).fillna(False)]
    if matches.empty:
        values = ", ".join(f"{column}={value}" for column, value in zip(stratify_cols, key, strict=True))
        raise ValueError(f"no selected confidence threshold for stratum: {values}")
    return float(matches["selected_margin_threshold"].iloc[0])


def _iter_strata(scores: pd.DataFrame, stratify_cols: Sequence[str]):
    for key, group in scores.groupby(list(stratify_cols), sort=False, dropna=False):
        yield key if isinstance(key, tuple) else (key,), group


def _prepend_stratum_values(frame: pd.DataFrame, stratify_cols: Sequence[str], key: tuple[object, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column, value in reversed(list(zip(stratify_cols, key, strict=True))):
        if column in out.columns:
            out[column] = value
        else:
            out.insert(0, column, value)
    return out


def _append_unique_context_columns(
    summary: pd.DataFrame, group: pd.DataFrame, stratify_cols: Sequence[str]
) -> pd.DataFrame:
    out = summary.copy()
    for column in ("matrix_id",):
        if column in stratify_cols or column not in group.columns:
            continue
        values = group[column].dropna().astype(str).unique()
        if len(values) == 1:
            out[column] = values[0]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
