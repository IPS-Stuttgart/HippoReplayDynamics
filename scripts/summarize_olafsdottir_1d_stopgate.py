#!/usr/bin/env python3
"""Summarize the Olafsdottir 1D pilot stopgate decision.

This script is deliberately non-scoring. It reads the existing pilot debug
reports and the cross-tier comparison pack, then writes a compact stopgate
artifact that records whether Olafsdottir should be scaled biologically or kept
as a technical portability/specificity result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from _provenance import build_script_provenance


REPORT_MANIFEST = "olafsdottir_1d_sleep_debug_report_manifest.json"
COMPARISON_INPUT = "olafsdottir_1d_pilot_tier_comparison.csv"
NORMALIZED_INPUT = "olafsdottir_1d_pilot_tier_normalized_margin_comparison.csv"
DECISION_INPUT = "olafsdottir_1d_pilot_tier_decision_summary.csv"

SUMMARY_OUTPUT = "olafsdottir_1d_stopgate_summary.csv"
GATE_OUTPUT = "olafsdottir_1d_stopgate_gate_summary.csv"
DECISION_OUTPUT = "olafsdottir_1d_stopgate_decision.md"
MANIFEST_OUTPUT = "olafsdottir_1d_stopgate_manifest.json"

REQUIRED_TIERS = ("balanced_debug", "high_information_debug", "high_information_holdout19_debug")


def run_stopgate_summary(
    *,
    balanced_report_dir: str | Path,
    high_information_report_dir: str | Path,
    holdout_report_dir: str | Path,
    comparison_dir: str | Path,
    output_dir: str | Path,
    margin_threshold: float = 5.5,
) -> dict[str, pd.DataFrame]:
    report_dirs = {
        "balanced_debug": Path(balanced_report_dir),
        "high_information_debug": Path(high_information_report_dir),
        "high_information_holdout19_debug": Path(holdout_report_dir),
    }
    comparison_root = Path(comparison_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    comparison = read_required_csv(comparison_root / COMPARISON_INPUT)
    normalized = read_required_csv(comparison_root / NORMALIZED_INPUT)
    tier_decision = read_required_csv(comparison_root / DECISION_INPUT)
    manifests = {label: read_optional_json(path / REPORT_MANIFEST) for label, path in report_dirs.items()}
    summary = build_stopgate_summary(
        comparison=comparison,
        normalized=normalized,
        tier_decision=tier_decision,
        manifests=manifests,
        margin_threshold=margin_threshold,
    )
    gates = build_gate_summary(summary.iloc[0])

    summary.to_csv(out / SUMMARY_OUTPUT, index=False)
    gates.to_csv(out / GATE_OUTPUT, index=False)
    (out / DECISION_OUTPUT).write_text(build_markdown_decision(summary.iloc[0], gates), encoding="utf-8")
    manifest = {
        "analysis": "olafsdottir_1d_stopgate",
        "output_dir": str(out),
        "margin_threshold": float(margin_threshold),
        **build_script_provenance(
            input_paths={
                "balanced_report_manifest": report_dirs["balanced_debug"] / REPORT_MANIFEST,
                "high_information_report_manifest": report_dirs["high_information_debug"] / REPORT_MANIFEST,
                "holdout_report_manifest": report_dirs["high_information_holdout19_debug"] / REPORT_MANIFEST,
                "pilot_tier_comparison": comparison_root / COMPARISON_INPUT,
                "pilot_tier_normalized_comparison": comparison_root / NORMALIZED_INPUT,
                "pilot_tier_decision": comparison_root / DECISION_INPUT,
            }
        ),
    }
    (out / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "gates": gates}


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_optional_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_stopgate_summary(
    *,
    comparison: pd.DataFrame,
    normalized: pd.DataFrame,
    tier_decision: pd.DataFrame,
    manifests: dict[str, dict[str, object]],
    margin_threshold: float,
) -> pd.DataFrame:
    missing_tiers = sorted(set(REQUIRED_TIERS).difference(set(comparison.get("tier_label", pd.Series(dtype=str)).astype(str))))
    tier_rows = {tier: row_for_tier(comparison, tier) for tier in REQUIRED_TIERS}
    balanced = tier_rows["balanced_debug"]
    high_info = tier_rows["high_information_debug"]
    holdout = tier_rows["high_information_holdout19_debug"]

    technical_scoreable = not missing_tiers and all(
        str(manifests.get(tier, {}).get("technical_classification", "technical-pass")) == "technical-pass"
        for tier in REQUIRED_TIERS
    )
    trajectory_family_supported = trajectory_supported(holdout, normalized, margin_threshold=margin_threshold)
    raw_trajectory_medians_negative = all(
        safe_float(tier_rows[tier], "median_delta_best_trajectory_minus_stationary") < 0 for tier in REQUIRED_TIERS
    )
    normalized_trajectory_medians_negative = all(
        normalized_median(normalized, tier, "trajectory_minus_stationary_per_second") < 0
        and normalized_median(normalized, tier, "trajectory_minus_stationary_per_spike") < 0
        for tier in REQUIRED_TIERS
    )
    imm_status = imm_fragmented_status(balanced, holdout, margin_threshold=margin_threshold)
    localized_signal = as_bool(tier_decision.iloc[0].get("localized_signal", False)) if not tier_decision.empty else False
    recommended = (
        "scale_biological_1d_pilot"
        if technical_scoreable and trajectory_family_supported
        else "continue_imm_fragmented_taxonomy_audit_only"
        if imm_status != "not_supported"
        else "stop_olafsdottir_biological_scaling_for_now"
    )
    forbid_pilot50_biology = not trajectory_family_supported
    row = {
        "technical_scoreable": bool(technical_scoreable),
        "trajectory_family_over_static_supported": bool(trajectory_family_supported),
        "imm_fragmented_axis_supported": imm_status,
        "localized_signal": localized_signal,
        "recommended_next_action": recommended,
        "forbid_pilot50_biology": bool(forbid_pilot50_biology),
        "missing_required_tiers": ";".join(missing_tiers),
        "balanced_events": value_or_nan(balanced, "events"),
        "high_information_events": value_or_nan(high_info, "events"),
        "holdout_events": value_or_nan(holdout, "events"),
        "balanced_trajectory_confident_events": value_or_nan(balanced, "trajectory_confident_events"),
        "high_information_trajectory_confident_events": value_or_nan(high_info, "trajectory_confident_events"),
        "holdout_trajectory_confident_events": value_or_nan(holdout, "trajectory_confident_events"),
        "balanced_median_trajectory_minus_stationary": value_or_nan(balanced, "median_delta_best_trajectory_minus_stationary"),
        "high_information_median_trajectory_minus_stationary": value_or_nan(high_info, "median_delta_best_trajectory_minus_stationary"),
        "holdout_median_trajectory_minus_stationary": value_or_nan(holdout, "median_delta_best_trajectory_minus_stationary"),
        "balanced_median_trajectory_per_second": normalized_median(normalized, "balanced_debug", "trajectory_minus_stationary_per_second"),
        "high_information_median_trajectory_per_second": normalized_median(normalized, "high_information_debug", "trajectory_minus_stationary_per_second"),
        "holdout_median_trajectory_per_second": normalized_median(normalized, "high_information_holdout19_debug", "trajectory_minus_stationary_per_second"),
        "balanced_median_trajectory_per_spike": normalized_median(normalized, "balanced_debug", "trajectory_minus_stationary_per_spike"),
        "high_information_median_trajectory_per_spike": normalized_median(normalized, "high_information_debug", "trajectory_minus_stationary_per_spike"),
        "holdout_median_trajectory_per_spike": normalized_median(normalized, "high_information_holdout19_debug", "trajectory_minus_stationary_per_spike"),
        "balanced_imm_confident_events": value_or_nan(balanced, "imm_confident_events"),
        "high_information_imm_confident_events": value_or_nan(high_info, "imm_confident_events"),
        "holdout_imm_confident_events": value_or_nan(holdout, "imm_confident_events"),
        "balanced_median_imm_minus_fragmented": value_or_nan(balanced, "median_delta_imm_minus_fragmented"),
        "high_information_median_imm_minus_fragmented": value_or_nan(high_info, "median_delta_imm_minus_fragmented"),
        "holdout_median_imm_minus_fragmented": value_or_nan(holdout, "median_delta_imm_minus_fragmented"),
        "raw_trajectory_medians_negative": bool(raw_trajectory_medians_negative),
        "normalized_trajectory_medians_negative": bool(normalized_trajectory_medians_negative),
        "pilot_tier_comparison_recommendation": tier_decision.iloc[0].get("recommendation", "") if not tier_decision.empty else "",
        "margin_threshold": float(margin_threshold),
        "claim_boundary": "technical portability and specificity result; no 1D trajectory-family biological scaling",
    }
    return pd.DataFrame([row])


def row_for_tier(comparison: pd.DataFrame, tier: str) -> pd.Series | None:
    if comparison.empty or "tier_label" not in comparison.columns:
        return None
    rows = comparison[comparison["tier_label"].astype(str).eq(tier)]
    return rows.iloc[0] if not rows.empty else None


def trajectory_supported(row: pd.Series | None, normalized: pd.DataFrame, *, margin_threshold: float) -> bool:
    if row is None:
        return False
    events = safe_float(row, "events")
    confident = safe_float(row, "trajectory_confident_events")
    raw_median = safe_float(row, "median_delta_best_trajectory_minus_stationary")
    per_second = normalized_median(normalized, "high_information_holdout19_debug", "trajectory_minus_stationary_per_second")
    per_spike = normalized_median(normalized, "high_information_holdout19_debug", "trajectory_minus_stationary_per_spike")
    return (
        events > 0
        and confident / events >= 0.5
        and raw_median > 0
        and per_second > 0
        and per_spike > 0
        and raw_median >= margin_threshold / 2
    )


def imm_fragmented_status(balanced: pd.Series | None, holdout: pd.Series | None, *, margin_threshold: float) -> str:
    if balanced is None or holdout is None:
        return "not_supported"
    holdout_median = safe_float(holdout, "median_delta_imm_minus_fragmented")
    balanced_median = safe_float(balanced, "median_delta_imm_minus_fragmented")
    holdout_confident = safe_float(holdout, "imm_confident_events")
    balanced_confident = safe_float(balanced, "imm_confident_events")
    holdout_events = safe_float(holdout, "events")
    if holdout_median >= margin_threshold and holdout_confident / max(holdout_events, 1.0) >= 0.5:
        return "supported"
    if holdout_median > 0 and (holdout_median > balanced_median or holdout_confident > balanced_confident):
        return "weak_or_partial"
    return "not_supported"


def normalized_median(normalized: pd.DataFrame, tier: str, margin: str) -> float:
    if normalized.empty:
        return np.nan
    rows = normalized[normalized["tier_label"].astype(str).eq(tier) & normalized["margin"].astype(str).eq(margin)]
    return float(rows["median_margin"].iloc[0]) if not rows.empty else np.nan


def safe_float(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return np.nan
    return float(row[column])


def value_or_nan(row: pd.Series | None, column: str) -> object:
    if row is None or column not in row:
        return np.nan
    return row[column]


def build_gate_summary(row: pd.Series) -> pd.DataFrame:
    gates = [
        gate(
            "technical_scoreable",
            bool(row["technical_scoreable"]),
            f"technical_scoreable={str(row['technical_scoreable']).lower()}",
            "all pilot tiers have technical-pass reports and comparison rows",
        ),
        gate(
            "trajectory_family_over_static_supported",
            bool(row["trajectory_family_over_static_supported"]),
            f"holdout_median={row['holdout_median_trajectory_minus_stationary']}; holdout_per_second={row['holdout_median_trajectory_per_second']}; holdout_per_spike={row['holdout_median_trajectory_per_spike']}",
            "holdout trajectory/static evidence must be positive in raw and normalized medians",
        ),
        gate(
            "raw_trajectory_medians_remain_negative",
            bool(row["raw_trajectory_medians_negative"]),
            f"balanced={row['balanced_median_trajectory_minus_stationary']}; high_information={row['high_information_median_trajectory_minus_stationary']}; holdout={row['holdout_median_trajectory_minus_stationary']}",
            "all debug pilot raw trajectory/static medians are negative",
        ),
        gate(
            "normalized_trajectory_medians_remain_negative",
            bool(row["normalized_trajectory_medians_negative"]),
            f"holdout_per_second={row['holdout_median_trajectory_per_second']}; holdout_per_spike={row['holdout_median_trajectory_per_spike']}",
            "normalized trajectory/static medians remain negative",
        ),
        gate(
            "imm_fragmented_axis_partial",
            str(row["imm_fragmented_axis_supported"]) in {"weak_or_partial", "supported"},
            f"status={row['imm_fragmented_axis_supported']}; holdout_median={row['holdout_median_imm_minus_fragmented']}",
            "IMM-vs-fragmented axis improves or remains positive but is not a trajectory-family claim",
        ),
        gate(
            "forbid_pilot50_biology",
            bool(row["forbid_pilot50_biology"]),
            f"forbid_pilot50_biology={str(row['forbid_pilot50_biology']).lower()}",
            "do not scale to pilot_50 as biology without trajectory/static support",
        ),
    ]
    return pd.DataFrame(gates)


def gate(name: str, passed: bool, value: str, requirement: str) -> dict[str, object]:
    return {
        "gate": name,
        "passed": bool(passed),
        "status": "pass" if passed else "fail",
        "value": value,
        "requirement": requirement,
    }


def build_markdown_decision(row: pd.Series, gates: pd.DataFrame) -> str:
    lines = [
        "# Olafsdottir 1D Stopgate Decision",
        "",
        "This stopgate reads existing debug reports and the pilot-tier comparison only. It does not rescore events.",
        "",
        "## Decision",
        "",
        f"- technical_scoreable: {str(row['technical_scoreable']).lower()}",
        f"- trajectory_family_over_static_supported: {str(row['trajectory_family_over_static_supported']).lower()}",
        f"- imm_fragmented_axis_supported: {row['imm_fragmented_axis_supported']}",
        f"- localized_signal: {str(row['localized_signal']).lower()}",
        f"- recommended_next_action: {row['recommended_next_action']}",
        f"- forbid_pilot50_biology: {str(row['forbid_pilot50_biology']).lower()}",
        "",
        "## Key Metrics",
        "",
        markdown_table(
            [
                ("balanced trajectory-confident", row["balanced_trajectory_confident_events"]),
                ("balanced median trajectory-stationary", row["balanced_median_trajectory_minus_stationary"]),
                ("high-information trajectory-confident", row["high_information_trajectory_confident_events"]),
                ("high-information median trajectory-stationary", row["high_information_median_trajectory_minus_stationary"]),
                ("holdout trajectory-confident", row["holdout_trajectory_confident_events"]),
                ("holdout median trajectory-stationary", row["holdout_median_trajectory_minus_stationary"]),
                ("holdout median trajectory per second", row["holdout_median_trajectory_per_second"]),
                ("holdout median trajectory per spike", row["holdout_median_trajectory_per_spike"]),
                ("holdout median IMM-fragmented", row["holdout_median_imm_minus_fragmented"]),
            ]
        ),
        "",
        "## Gate Summary",
        "",
        dataframe_to_markdown(gates[["gate", "status", "value"]]),
        "",
        "## Claim Boundary",
        "",
        "Olafsdottir 1D is a technical portability and specificity result at this stage. It should not be used as a cross-dataset trajectory-family replication or as a 1D-vs-2D biological comparison.",
        "",
    ]
    return "\n".join(lines)


def markdown_table(rows: Sequence[tuple[object, object]]) -> str:
    table = ["| Metric | Value |", "| --- | --- |"]
    table.extend(f"| {key} | {format_value(value)} |" for key, value in rows)
    return "\n".join(table)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    table = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        table.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(table)


def format_value(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-report-dir", type=Path, required=True)
    parser.add_argument("--high-information-report-dir", type=Path, required=True)
    parser.add_argument("--holdout-report-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tables = run_stopgate_summary(
        balanced_report_dir=args.balanced_report_dir,
        high_information_report_dir=args.high_information_report_dir,
        holdout_report_dir=args.holdout_report_dir,
        comparison_dir=args.comparison_dir,
        output_dir=args.output_dir,
        margin_threshold=args.margin_threshold,
    )
    print(tables["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
