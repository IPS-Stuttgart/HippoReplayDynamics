#!/usr/bin/env python3
"""Select a synthetic-calibrated momentum confidence threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

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
DEFAULT_THRESHOLDS = tuple(float(value) for value in range(0, 11))
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
    parser.add_argument("--max-false-positive-claims", type=int, default=0)
    parser.add_argument("--min-positive-claim-recall", type=float, default=0.0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_float_values(args.thresholds)
    recovery_scores_path = find_score_file(args.recovery_scores)
    recovery = pd.read_csv(recovery_scores_path)
    recovery_group_cols = _parse_group_cols(args.recovery_group_cols) or infer_paired_model_group_cols(recovery)
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
    recovery_sweep.to_csv(out / "momentum_confidence_threshold_recovery_sweep.csv", index=False)
    selection.to_csv(out / "momentum_confidence_threshold_selection.csv", index=False)
    _summaries_by_stratum(
        recovery,
        stratum_col=args.stratify_column,
        threshold=selected_threshold,
        positive_model=args.positive_model,
        reference_model=args.reference_model,
        true_model_col=args.true_model_column,
        positive_true_label=args.positive_true_label,
    ).to_csv(out / "momentum_confidence_threshold_recovery_by_stratum.csv", index=False)

    print(selection.to_string(index=False))

    if args.evidence_scores:
        evidence_scores_path = find_score_file(args.evidence_scores)
        evidence = pd.read_csv(evidence_scores_path)
        evidence_group_cols = _parse_group_cols(args.evidence_group_cols) or infer_paired_model_group_cols(evidence)
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
        evidence_summary = paired_model_margin_summary(evidence_decisions)
        evidence_summary["selected_margin_threshold"] = selected_threshold
        evidence_summary["recovery_scores"] = str(recovery_scores_path)
        evidence_summary["evidence_scores"] = str(evidence_scores_path)
        evidence_summary["recovery_group_cols"] = ",".join(recovery_group_cols)
        evidence_summary["evidence_group_cols"] = ",".join(evidence_group_cols)
        evidence_sweep.to_csv(out / "momentum_confidence_threshold_evidence_sweep.csv", index=False)
        evidence_decisions.to_csv(out / "momentum_confidence_threshold_evidence_decisions.csv", index=False)
        evidence_summary.to_csv(out / "momentum_confidence_threshold_evidence_summary.csv", index=False)
        _summaries_by_stratum(
            evidence,
            stratum_col=args.stratify_column,
            threshold=selected_threshold,
            positive_model=args.positive_model,
            reference_model=args.reference_model,
        ).to_csv(out / "momentum_confidence_threshold_evidence_by_stratum.csv", index=False)
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


def _summaries_by_stratum(
    scores: pd.DataFrame,
    *,
    stratum_col: str,
    threshold: float,
    positive_model: str,
    reference_model: str,
    true_model_col: str | None = None,
    positive_true_label: str | None = None,
) -> pd.DataFrame:
    if not stratum_col or stratum_col not in scores.columns:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for value, group in scores.groupby(stratum_col, sort=False):
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
        summary.insert(0, stratum_col, value)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
