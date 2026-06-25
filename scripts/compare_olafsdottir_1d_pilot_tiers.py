#!/usr/bin/env python3
"""Compare Olafsdottir 1D SleepPOST pilot evidence debug tiers.

This helper is deliberately non-scoring. It reads existing debug report outputs
from ``report_olafsdottir_1d_sleeppost_evidence.py`` and compares raw margins
against per-second and per-spike normalized margins so event-duration effects do
not masquerade as biological signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from _provenance import build_script_provenance

QUALITY_INPUT = "olafsdottir_1d_sleep_debug_quality_table.csv"
REPORT_MANIFEST_INPUT = "olafsdottir_1d_sleep_debug_report_manifest.json"

COMPARISON_OUTPUT = "olafsdottir_1d_pilot_tier_comparison.csv"
NORMALIZED_OUTPUT = "olafsdottir_1d_pilot_tier_normalized_margin_comparison.csv"
BY_ANIMAL_OUTPUT = "olafsdottir_1d_pilot_tier_by_animal_comparison.csv"
BY_PAIR_OUTPUT = "olafsdottir_1d_pilot_tier_by_pair_comparison.csv"
DECISION_OUTPUT = "olafsdottir_1d_pilot_tier_decision_summary.csv"
REPORT_OUTPUT = "olafsdottir_1d_pilot_tier_comparison.md"
MANIFEST_OUTPUT = "olafsdottir_1d_pilot_tier_comparison_manifest.json"

DEFAULT_LABELS = ("balanced_debug", "high_information_debug", "high_information_holdout19_debug")
PAIR_KEYS = ["animal", "date", "track1_session", "sleeppost_session"]
MARGIN_COLUMNS = [
    "delta_best_trajectory_minus_stationary",
    "delta_imm_minus_fragmented",
    "trajectory_minus_stationary_per_second",
    "trajectory_minus_stationary_per_spike",
    "imm_minus_fragmented_per_second",
    "imm_minus_fragmented_per_spike",
]
SUMMARY_COLUMNS = [
    "tier_label",
    "pilot_tier",
    "events",
    "animals",
    "pairs",
    "trajectory_confident_events",
    "nontrajectory_confident_events",
    "imm_confident_events",
    "fragmented_confident_events",
    "positive_trajectory_margin_events",
    "positive_imm_margin_events",
    "mean_delta_best_trajectory_minus_stationary",
    "median_delta_best_trajectory_minus_stationary",
    "mean_delta_imm_minus_fragmented",
    "median_delta_imm_minus_fragmented",
    "mean_trajectory_minus_stationary_per_second",
    "median_trajectory_minus_stationary_per_second",
    "mean_trajectory_minus_stationary_per_spike",
    "median_trajectory_minus_stationary_per_spike",
    "mean_imm_minus_fragmented_per_second",
    "median_imm_minus_fragmented_per_second",
    "mean_imm_minus_fragmented_per_spike",
    "median_imm_minus_fragmented_per_spike",
    "mean_duration_ms",
    "median_duration_ms",
    "mean_n_spikes",
    "median_n_spikes",
    "biological_claim_assessed",
]


def run_pilot_tier_comparison(
    *,
    report_dirs: Sequence[str | Path],
    labels: Sequence[str] | None = None,
    output_dir: str | Path,
    margin_threshold: float = 5.5,
    event_qc: str | Path | None = None,
    decoder_qc: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    labels = tuple(labels or default_labels(len(report_dirs)))
    if len(labels) != len(report_dirs):
        raise ValueError("labels must have the same length as report_dirs")
    if not report_dirs:
        raise ValueError("at least one report directory is required")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    quality = load_all_quality_tables(report_dirs, labels)
    quality = add_normalized_margins(quality)
    comparison = build_tier_comparison(quality, margin_threshold=margin_threshold)
    normalized = build_normalized_margin_comparison(quality)
    by_animal = build_group_comparison(quality, ["animal"], margin_threshold=margin_threshold)
    by_pair = build_group_comparison(quality, PAIR_KEYS, margin_threshold=margin_threshold)
    decision = build_decision_summary(comparison, normalized, margin_threshold=margin_threshold)

    comparison.to_csv(out / COMPARISON_OUTPUT, index=False)
    normalized.to_csv(out / NORMALIZED_OUTPUT, index=False)
    by_animal.to_csv(out / BY_ANIMAL_OUTPUT, index=False)
    by_pair.to_csv(out / BY_PAIR_OUTPUT, index=False)
    decision.to_csv(out / DECISION_OUTPUT, index=False)
    (out / REPORT_OUTPUT).write_text(
        build_markdown_report(
            comparison=comparison,
            normalized=normalized,
            by_animal=by_animal,
            by_pair=by_pair,
            decision=decision,
            margin_threshold=margin_threshold,
        ),
        encoding="utf-8",
    )
    manifest = {
        "analysis": "olafsdottir_1d_pilot_tier_comparison",
        "report_dirs": [str(Path(path)) for path in report_dirs],
        "labels": list(labels),
        "output_dir": str(out),
        "margin_threshold": float(margin_threshold),
        **build_script_provenance(
            input_paths={
                **{f"report_{label}": Path(path) / QUALITY_INPUT for label, path in zip(labels, report_dirs, strict=True)},
                **optional_input_paths(event_qc=event_qc, decoder_qc=decoder_qc),
            }
        ),
    }
    (out / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "quality": quality,
        "comparison": comparison,
        "normalized": normalized,
        "by_animal": by_animal,
        "by_pair": by_pair,
        "decision": decision,
    }


def default_labels(n: int) -> tuple[str, ...]:
    if n <= len(DEFAULT_LABELS):
        return DEFAULT_LABELS[:n]
    return tuple([*DEFAULT_LABELS, *[f"tier_{index + 1}" for index in range(len(DEFAULT_LABELS), n)]])


def optional_input_paths(*, event_qc: str | Path | None, decoder_qc: str | Path | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if event_qc is not None:
        paths["event_qc"] = Path(event_qc)
    if decoder_qc is not None:
        paths["decoder_qc"] = Path(decoder_qc)
    return paths


def load_all_quality_tables(report_dirs: Sequence[str | Path], labels: Sequence[str]) -> pd.DataFrame:
    frames = []
    for label, report_dir in zip(labels, report_dirs, strict=True):
        root = Path(report_dir)
        path = root / QUALITY_INPUT
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "tier_label", str(label))
        frame.insert(1, "report_dir", str(root))
        manifest = read_manifest(root / REPORT_MANIFEST_INPUT)
        if "pilot_tier" not in frame.columns or frame["pilot_tier"].isna().all():
            frame["pilot_tier"] = manifest.get("source_pilot_tier", label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def read_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def add_normalized_margins(quality: pd.DataFrame) -> pd.DataFrame:
    quality = quality.copy()
    duration_s = pd.to_numeric(quality.get("duration_ms"), errors="coerce") / 1000.0
    n_spikes = pd.to_numeric(quality.get("n_spikes"), errors="coerce")
    trajectory = pd.to_numeric(quality.get("delta_best_trajectory_minus_stationary"), errors="coerce")
    imm = pd.to_numeric(quality.get("delta_imm_minus_fragmented"), errors="coerce")
    quality["duration_s"] = duration_s
    quality["trajectory_minus_stationary_per_second"] = safe_divide(trajectory, duration_s)
    quality["trajectory_minus_stationary_per_spike"] = safe_divide(trajectory, n_spikes)
    quality["imm_minus_fragmented_per_second"] = safe_divide(imm, duration_s)
    quality["imm_minus_fragmented_per_spike"] = safe_divide(imm, n_spikes)
    return quality


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator > 0)
    return numerator / denominator


def build_tier_comparison(quality: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    rows = []
    for tier_label, group in quality.groupby("tier_label", sort=False):
        rows.append(summary_row(group, tier_label=tier_label, margin_threshold=margin_threshold))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summary_row(group: pd.DataFrame, *, tier_label: str, margin_threshold: float) -> dict[str, object]:
    trajectory = pd.to_numeric(group.get("delta_best_trajectory_minus_stationary"), errors="coerce")
    imm = pd.to_numeric(group.get("delta_imm_minus_fragmented"), errors="coerce")
    row = {
        "tier_label": tier_label,
        "pilot_tier": common_value(group, "pilot_tier"),
        "events": int(len(group)),
        "animals": int(group["animal"].nunique()) if "animal" in group else 0,
        "pairs": int(group[PAIR_KEYS].drop_duplicates().shape[0]) if set(PAIR_KEYS).issubset(group.columns) else 0,
        "trajectory_confident_events": int(is_trajectory_confident(group).sum()),
        "nontrajectory_confident_events": int(is_nontrajectory_confident(group).sum()),
        "imm_confident_events": int(as_bool_series(group.get("imm_clean_vs_fragmented_claim")).sum()),
        "fragmented_confident_events": int(as_bool_series(group.get("fragmented_claim")).sum()),
        "positive_trajectory_margin_events": int(trajectory.gt(0).sum()),
        "positive_imm_margin_events": int(imm.gt(0).sum()),
        "mean_delta_best_trajectory_minus_stationary": finite_mean(trajectory),
        "median_delta_best_trajectory_minus_stationary": finite_median(trajectory),
        "mean_delta_imm_minus_fragmented": finite_mean(imm),
        "median_delta_imm_minus_fragmented": finite_median(imm),
        "mean_duration_ms": finite_mean(pd.to_numeric(group.get("duration_ms"), errors="coerce")),
        "median_duration_ms": finite_median(pd.to_numeric(group.get("duration_ms"), errors="coerce")),
        "mean_n_spikes": finite_mean(pd.to_numeric(group.get("n_spikes"), errors="coerce")),
        "median_n_spikes": finite_median(pd.to_numeric(group.get("n_spikes"), errors="coerce")),
        "biological_claim_assessed": False,
    }
    for column in [
        "trajectory_minus_stationary_per_second",
        "trajectory_minus_stationary_per_spike",
        "imm_minus_fragmented_per_second",
        "imm_minus_fragmented_per_spike",
    ]:
        values = pd.to_numeric(group.get(column), errors="coerce")
        row[f"mean_{column}"] = finite_mean(values)
        row[f"median_{column}"] = finite_median(values)
    return row


def build_normalized_margin_comparison(quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier_label, group in quality.groupby("tier_label", sort=False):
        for margin in MARGIN_COLUMNS:
            values = pd.to_numeric(group.get(margin), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "tier_label": tier_label,
                    "margin": margin,
                    "events_with_finite_margin": int(values.shape[0]),
                    "mean_margin": finite_mean(values),
                    "median_margin": finite_median(values),
                    "p25_margin": finite_quantile(values, 0.25),
                    "p75_margin": finite_quantile(values, 0.75),
                    "positive_margin_events": int(values.gt(0).sum()),
                    "positive_margin_fraction": float(values.gt(0).mean()) if not values.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_group_comparison(quality: pd.DataFrame, group_keys: Sequence[str], *, margin_threshold: float) -> pd.DataFrame:
    if not set(group_keys).issubset(quality.columns):
        return pd.DataFrame()
    rows = []
    for keys, group in quality.groupby(["tier_label", *group_keys], dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tier_label = str(keys[0])
        key_values = dict(zip(group_keys, keys[1:], strict=True))
        rows.append({**key_values, **summary_row(group, tier_label=tier_label, margin_threshold=margin_threshold)})
    key_columns = list(group_keys)
    summary_columns = [column for column in SUMMARY_COLUMNS if column not in key_columns]
    return pd.DataFrame(rows, columns=[*key_columns, *summary_columns])


def build_decision_summary(comparison: pd.DataFrame, normalized: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    labels = set(comparison["tier_label"].astype(str)) if not comparison.empty else set()
    balanced = row_for_label(comparison, "balanced_debug")
    nonholdout = row_for_label(comparison, "high_information_debug")
    holdout = row_for_label(comparison, "high_information_holdout19_debug")
    recommendation = "insufficient_tiers"
    reason = "Required balanced, high-information, and holdout tiers were not all available."
    if balanced is not None and holdout is not None:
        trajectory_raw_improved = float(holdout["median_delta_best_trajectory_minus_stationary"]) > float(
            balanced["median_delta_best_trajectory_minus_stationary"]
        )
        trajectory_count_improved = int(holdout["trajectory_confident_events"]) > int(balanced["trajectory_confident_events"])
        trajectory_normalized_improved = normalized_median(normalized, "high_information_holdout19_debug", "trajectory_minus_stationary_per_second") > normalized_median(
            normalized, "balanced_debug", "trajectory_minus_stationary_per_second"
        )
        imm_raw_improved = float(holdout["median_delta_imm_minus_fragmented"]) > float(balanced["median_delta_imm_minus_fragmented"])
        imm_count_improved = int(holdout["imm_confident_events"]) > int(balanced["imm_confident_events"])
        if trajectory_raw_improved and trajectory_count_improved and trajectory_normalized_improved:
            recommendation = "define_frozen_high_information_confirmation_tier"
            reason = "Holdout improves trajectory/static raw, normalized, and confident-count readouts versus balanced debug."
        elif imm_raw_improved and imm_count_improved and not trajectory_raw_improved:
            recommendation = "continue_imm_fragmented_taxonomy_audit_only"
            reason = "Holdout improves IMM-vs-fragmented but not median trajectory-vs-stationary evidence."
        elif trajectory_count_improved and not trajectory_raw_improved:
            recommendation = "do_not_scale_biological_claim"
            reason = "Holdout increases trajectory-confident count but median trajectory-vs-stationary remains weaker than balanced debug."
        else:
            recommendation = "stop_biological_scaling_for_now"
            reason = "Holdout does not improve the trajectory-family readout versus balanced debug."
    return pd.DataFrame(
        [
            {
                "recommendation": recommendation,
                "reason": reason,
                "available_tiers": ";".join(sorted(labels)),
                "margin_threshold": float(margin_threshold),
                "balanced_median_trajectory_minus_stationary": value_or_nan(balanced, "median_delta_best_trajectory_minus_stationary"),
                "holdout_median_trajectory_minus_stationary": value_or_nan(holdout, "median_delta_best_trajectory_minus_stationary"),
                "balanced_trajectory_confident_events": value_or_nan(balanced, "trajectory_confident_events"),
                "holdout_trajectory_confident_events": value_or_nan(holdout, "trajectory_confident_events"),
                "balanced_median_imm_minus_fragmented": value_or_nan(balanced, "median_delta_imm_minus_fragmented"),
                "holdout_median_imm_minus_fragmented": value_or_nan(holdout, "median_delta_imm_minus_fragmented"),
                "balanced_imm_confident_events": value_or_nan(balanced, "imm_confident_events"),
                "holdout_imm_confident_events": value_or_nan(holdout, "imm_confident_events"),
                "balanced_median_trajectory_per_second": normalized_median(normalized, "balanced_debug", "trajectory_minus_stationary_per_second"),
                "holdout_median_trajectory_per_second": normalized_median(normalized, "high_information_holdout19_debug", "trajectory_minus_stationary_per_second"),
                "balanced_median_trajectory_per_spike": normalized_median(normalized, "balanced_debug", "trajectory_minus_stationary_per_spike"),
                "holdout_median_trajectory_per_spike": normalized_median(normalized, "high_information_holdout19_debug", "trajectory_minus_stationary_per_spike"),
                "nonholdout_median_trajectory_minus_stationary": value_or_nan(nonholdout, "median_delta_best_trajectory_minus_stationary"),
                "nonholdout_median_imm_minus_fragmented": value_or_nan(nonholdout, "median_delta_imm_minus_fragmented"),
                "biological_claim_assessed": False,
            }
        ]
    )


def row_for_label(frame: pd.DataFrame, label: str) -> pd.Series | None:
    if frame.empty or "tier_label" not in frame:
        return None
    rows = frame[frame["tier_label"].astype(str).eq(label)]
    return rows.iloc[0] if not rows.empty else None


def normalized_median(normalized: pd.DataFrame, label: str, margin: str) -> float:
    if normalized.empty:
        return np.nan
    rows = normalized[normalized["tier_label"].astype(str).eq(label) & normalized["margin"].astype(str).eq(margin)]
    return float(rows["median_margin"].iloc[0]) if not rows.empty else np.nan


def value_or_nan(row: pd.Series | None, column: str) -> object:
    if row is None or column not in row:
        return np.nan
    return row[column]


def common_value(group: pd.DataFrame, column: str) -> object:
    if column not in group:
        return ""
    values = group[column].dropna().astype(str).unique()
    return values[0] if len(values) == 1 else ";".join(sorted(values))


def finite_mean(values: pd.Series) -> float:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else np.nan


def finite_median(values: pd.Series) -> float:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else np.nan


def finite_quantile(values: pd.Series, q: float) -> float:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.quantile(q)) if not values.empty else np.nan


def is_trajectory_confident(group: pd.DataFrame) -> pd.Series:
    claim = group.get("trajectory_family_claim", pd.Series(index=group.index, dtype=object)).astype(str).str.lower()
    return claim.eq("trajectory_confident") | claim.eq("trajectory")


def is_nontrajectory_confident(group: pd.DataFrame) -> pd.Series:
    claim = group.get("trajectory_family_claim", pd.Series(index=group.index, dtype=object)).astype(str).str.lower()
    return claim.eq("nontrajectory_confident") | claim.eq("stationary_confident") | claim.eq("nontrajectory")


def as_bool_series(values: object) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    series = pd.Series(values)
    return series.map(as_bool)


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass"}


def build_markdown_report(
    *,
    comparison: pd.DataFrame,
    normalized: pd.DataFrame,
    by_animal: pd.DataFrame,
    by_pair: pd.DataFrame,
    decision: pd.DataFrame,
    margin_threshold: float,
) -> str:
    decision_row = decision.iloc[0].to_dict() if not decision.empty else {}
    lines = [
        "# Olafsdottir 1D Pilot Tier Comparison",
        "",
        "This comparison reads existing debug reports only. It does not select events and does not rescore evidence.",
        "",
        "## Decision",
        "",
        f"- recommendation: {decision_row.get('recommendation', '')}",
        f"- reason: {decision_row.get('reason', '')}",
        f"- margin threshold: {margin_threshold:g}",
        "- biological_claim_assessed: false",
        "",
        "## Tier Summary",
        "",
        markdown_table(
            comparison[
                [
                    "tier_label",
                    "events",
                    "trajectory_confident_events",
                    "median_delta_best_trajectory_minus_stationary",
                    "imm_confident_events",
                    "median_delta_imm_minus_fragmented",
                    "median_trajectory_minus_stationary_per_second",
                    "median_trajectory_minus_stationary_per_spike",
                ]
            ]
        ),
        "",
        "## Normalized Margin Summary",
        "",
        markdown_table(
            normalized[
                normalized["margin"].isin(
                    [
                        "trajectory_minus_stationary_per_second",
                        "trajectory_minus_stationary_per_spike",
                        "imm_minus_fragmented_per_second",
                        "imm_minus_fragmented_per_spike",
                    ]
                )
            ][["tier_label", "margin", "median_margin", "positive_margin_events", "positive_margin_fraction"]]
        ),
        "",
        "## Animal Summary",
        "",
        markdown_table(
            by_animal[
                [
                    "tier_label",
                    "animal",
                    "events",
                    "trajectory_confident_events",
                    "median_delta_best_trajectory_minus_stationary",
                    "imm_confident_events",
                    "median_delta_imm_minus_fragmented",
                ]
            ]
            if not by_animal.empty
            else by_animal
        ),
        "",
        "## Claim Boundary",
        "",
        "Do not treat this output as a 1D-vs-2D comparison or cross-dataset biological claim. It only decides whether the debug tiers justify another frozen diagnostic selection.",
        "",
    ]
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    rows = [[str(column) for column in frame.columns]]
    rows.append(["---"] * len(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        rows.append([format_value(value) for value in row])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def format_value(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def parse_labelled_report_dirs(values: Sequence[str]) -> tuple[list[str], list[Path]]:
    labels: list[str] = []
    paths: list[Path] = []
    for index, value in enumerate(values):
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            label = default_labels(len(values))[index]
            path = value
        labels.append(label)
        paths.append(Path(path))
    return labels, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        action="append",
        required=True,
        help="Report directory, optionally labelled as label=/path/to/report. Pass once per tier.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-qc", type=Path, default=None, help="Optional event QC path recorded in provenance only.")
    parser.add_argument("--decoder-qc", type=Path, default=None, help="Optional decoder QC path recorded in provenance only.")
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels, report_dirs = parse_labelled_report_dirs(args.report_dir)
    tables = run_pilot_tier_comparison(
        report_dirs=report_dirs,
        labels=labels,
        output_dir=args.output_dir,
        margin_threshold=args.margin_threshold,
        event_qc=args.event_qc,
        decoder_qc=args.decoder_qc,
    )
    print(tables["decision"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
