#!/usr/bin/env python3
"""Write post-hoc replay model-evidence result-quality audit outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hipporeplayimm.result_quality_audit import (
    ObservationCalibrationSelectionConfig,
    write_result_quality_audit,
)


def _optional_frame(path: str | None) -> pd.DataFrame | None:
    if path is None or not str(path).strip():
        return None
    return pd.read_csv(path)


def _optional_json(path: str | None) -> dict[str, object] | None:
    if path is None or not str(path).strip():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit replay model-evidence CSV outputs.")
    parser.add_argument("--scores", required=True, help="Path to event_model_evidence.csv or event_scores.csv.")
    parser.add_argument("--output", required=True, help="Directory for audit outputs.")
    parser.add_argument(
        "--common-support-scores",
        help="Optional score table from a common-support diagnostic rerun.",
    )
    parser.add_argument(
        "--observation-sweep-summary",
        help="Optional observation_sweep_summary.csv to select validation-gated calibration settings.",
    )
    parser.add_argument("--max-behavior-error-cm", type=float)
    parser.add_argument("--min-recovery-accuracy", type=float)
    parser.add_argument(
        "--allow-real-evidence-selected",
        action="store_true",
        help="Allow calibration rows marked as selected using real replay evidence.",
    )
    parser.add_argument("--top-calibrations", type=int, default=10)
    parser.add_argument(
        "--provenance-json",
        help="Optional JSON object matching ProvenanceRecord fields.",
    )
    args = parser.parse_args()

    selection = ObservationCalibrationSelectionConfig(
        max_behavior_error_cm=args.max_behavior_error_cm,
        min_recovery_accuracy=args.min_recovery_accuracy,
        forbid_real_evidence_selected=not args.allow_real_evidence_selected,
        top_k=args.top_calibrations,
    )
    dashboard = write_result_quality_audit(
        pd.read_csv(args.scores),
        args.output,
        common_support_scores=_optional_frame(args.common_support_scores),
        observation_sweep_summary=_optional_frame(args.observation_sweep_summary),
        observation_selection_config=selection,
        provenance=_optional_json(args.provenance_json),
    )
    print(f"Wrote result-quality audit dashboard: {dashboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
