#!/usr/bin/env python3
"""Create advanced diagnostics for replay model-evidence CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    ProvenanceRecord,
    add_evidence_margin_columns,
    common_support_audit,
    evidence_margin_table,
    hierarchical_bootstrap,
    hierarchical_summary,
    leave_one_group_influence,
    model_disagreement_events,
    paired_model_margin_decisions,
    paired_model_margin_summary,
    provenance_audit,
    wrong_map_delta_summary,
    write_dashboard,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="event_model_evidence.csv from a benchmark run")
    parser.add_argument("--output", default="results/advanced-diagnostics")
    parser.add_argument("--wrong-map-scores", help="Optional wrong-environment-map event_model_evidence.csv")
    parser.add_argument("--common-support-scores", help="Optional common-support diagnostic event_model_evidence.csv")
    parser.add_argument("--bootstrap-model", action="append", default=[], help="Model name for event/session/rat bootstrap summaries; can repeat")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--parameter-source", default="unknown", choices=["unknown", "default", "synthetic_selected", "real_selected", "manual"])
    parser.add_argument("--selection-run-id", default="")
    parser.add_argument("--selection-metric", default="")
    parser.add_argument("--selection-passed-recovery-gate", choices=["true", "false", "unknown"], default="unknown")
    parser.add_argument("--selection-used-real-evidence", choices=["true", "false", "unknown"], default="unknown")
    parser.add_argument("--notes", default="")
    parser.add_argument("--paired-positive-model", default="", help="Optional model to test as a margin-gated positive claim.")
    parser.add_argument("--paired-reference-model", default="", help="Optional reference model for paired margin-gated decisions.")
    parser.add_argument("--paired-margin-threshold", type=float, default=0.0)
    parser.add_argument(
        "--paired-group-cols",
        default="session,event_index",
        help="Comma-separated columns defining one paired decision row.",
    )
    parser.add_argument("--paired-true-model-column", default="")
    parser.add_argument("--paired-positive-true-label", default="")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def parse_optional_bool(value: str):
        if value == "unknown":
            return None
        return value == "true"

    provenance = ProvenanceRecord(
        parameter_source=args.parameter_source,
        selection_run_id=args.selection_run_id,
        selection_metric=args.selection_metric,
        selection_passed_recovery_gate=parse_optional_bool(args.selection_passed_recovery_gate),
        selection_used_real_evidence=parse_optional_bool(args.selection_used_real_evidence),
        notes=args.notes,
    )

    add_evidence_margin_columns(scores).to_csv(out / "event_scores_with_margins.csv", index=False)
    evidence_margin_table(scores).to_csv(out / "evidence_margins.csv", index=False)
    hierarchical_summary(scores).to_csv(out / "hierarchical_summary.csv", index=False)
    model_disagreement_events(scores).to_csv(out / "model_disagreement_events.csv", index=False)
    provenance_audit(scores, provenance).to_csv(out / "provenance_audit.csv", index=False)

    if args.paired_positive_model and args.paired_reference_model:
        paired_group_cols = tuple(col.strip() for col in args.paired_group_cols.split(",") if col.strip())
        paired_decisions = paired_model_margin_decisions(
            scores,
            positive_model=args.paired_positive_model,
            reference_model=args.paired_reference_model,
            margin_threshold=args.paired_margin_threshold,
            group_cols=paired_group_cols,
            true_model_col=args.paired_true_model_column or None,
            positive_true_label=args.paired_positive_true_label or None,
        )
        paired_decisions.to_csv(out / "paired_model_margin_decisions.csv", index=False)
        paired_model_margin_summary(
            paired_decisions,
            true_model_col=args.paired_true_model_column or None,
        ).to_csv(out / "paired_model_margin_summary.csv", index=False)

    if "session" in scores.columns:
        leave_one_group_influence(scores, group_col="session").to_csv(out / "leave_one_session_influence.csv", index=False)
        rat_scores = scores.copy()
        rat_scores["rat"] = rat_scores["session"].astype(str).str.split("/", n=1).str[0]
        leave_one_group_influence(rat_scores, group_col="rat").to_csv(out / "leave_one_rat_influence.csv", index=False)

    if args.bootstrap_model:
        rows = []
        for model in args.bootstrap_model:
            for level in ("event", "session", "rat"):
                rows.append(
                    hierarchical_bootstrap(
                        scores,
                        model=model,
                        level=level,
                        n_bootstrap=args.bootstrap_samples,
                        random_seed=args.random_seed,
                    )
                )
        pd.DataFrame(rows).to_csv(out / "hierarchical_bootstrap.csv", index=False)

    if args.wrong_map_scores:
        wrong = pd.read_csv(args.wrong_map_scores)
        wrong_map_delta_summary(scores, wrong).to_csv(out / "wrong_map_delta_summary.csv", index=False)

    if args.common_support_scores:
        common = pd.read_csv(args.common_support_scores)
        common_support_audit(scores, common).to_csv(out / "common_support_audit.csv", index=False)

    dashboard = write_dashboard(scores, out, provenance=provenance)
    print(f"Wrote diagnostics to {out}")
    print(f"Dashboard: {dashboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
