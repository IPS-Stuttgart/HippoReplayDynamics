#!/usr/bin/env python3
"""Triage Olafsdottir Track1 decoder QC failures and threshold sensitivity."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


FAILURE_OUTPUT = "olafsdottir_track1_decoder_failure_reason_summary.csv"
SENSITIVITY_OUTPUT = "olafsdottir_track1_decoder_threshold_sensitivity.csv"
METRIC_OUTPUT = "olafsdottir_track1_decoder_metric_distribution.csv"
PAIR_AUDIT_OUTPUT = "olafsdottir_track1_decoder_pair_status_audit.csv"
GATE_OUTPUT = "olafsdottir_track1_decoder_qc_triage_gate_summary.csv"
SUMMARY_OUTPUT = "olafsdottir_track1_decoder_qc_triage_summary.md"

DEFAULT_MIN_UNITS = 5
DEFAULT_MAX_POSTERIOR_ERROR_CM = 35.0
DEFAULT_MAX_MAP_ERROR_CM = 45.0
DEFAULT_MIN_POSTERIOR_COVERAGE = 0.80

MIN_UNITS_GRID = (3, 5, 8, 10)
POSTERIOR_ERROR_GRID = (25.0, 35.0, 50.0, 75.0, 100.0)
MAP_ERROR_GRID = (35.0, 45.0, 75.0, 100.0, 150.0)
COVERAGE_GRID = (0.50, 0.65, 0.80)

REQUIRED_DECODER_COLUMNS = {
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "decoder_status",
    "encoding_units_passing_qc",
    "posterior_mean_error_cm_median",
    "map_error_cm_median",
    "posterior_coverage_fraction",
}
REQUIRED_PAIR_COLUMNS = {
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
}
REQUIRED_UNIT_COLUMNS = {
    "animal",
    "date",
    "track1_session",
}

PAIR_AUDIT_COLUMNS = [
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "decoder_qc_status",
    "decoder_qc_passed",
    "encoding_units_passing_qc",
    "posterior_mean_error_cm_median",
    "map_error_cm_median",
    "posterior_coverage_fraction",
    "track_spike_position_overlap_s",
    "failed_min_units",
    "failed_posterior_mean_error",
    "failed_map_error",
    "failed_posterior_coverage",
    "failed_missing_spike_position_overlap",
    "failed_missing_metrics",
    "failed_schema",
    "threshold_expected_passed",
    "primary_failure_reason",
]

FAILURE_REASONS = [
    ("too_few_encoding_units", "failed_min_units"),
    ("poor_posterior_mean_error", "failed_posterior_mean_error"),
    ("poor_map_error", "failed_map_error"),
    ("low_posterior_coverage", "failed_posterior_coverage"),
    ("missing_spike_position_overlap", "failed_missing_spike_position_overlap"),
    ("missing_finite_decoder_metrics", "failed_missing_metrics"),
    ("schema_status_mismatch", "failed_schema"),
]


def run_decoder_qc_triage(
    *,
    decoder_qc: str | Path,
    unit_qc: str | Path,
    pairs_csv: str | Path,
    output_dir: str | Path,
    min_units: int = DEFAULT_MIN_UNITS,
    max_posterior_mean_error_cm_median: float = DEFAULT_MAX_POSTERIOR_ERROR_CM,
    max_map_error_cm_median: float = DEFAULT_MAX_MAP_ERROR_CM,
    min_posterior_coverage_fraction: float = DEFAULT_MIN_POSTERIOR_COVERAGE,
    min_units_grid: Sequence[int] = MIN_UNITS_GRID,
    posterior_error_grid: Sequence[float] = POSTERIOR_ERROR_GRID,
    map_error_grid: Sequence[float] = MAP_ERROR_GRID,
    coverage_grid: Sequence[float] = COVERAGE_GRID,
) -> dict[str, pd.DataFrame]:
    decoder = load_decoder_qc(decoder_qc)
    units = load_unit_qc(unit_qc)
    pairs = load_pairs(pairs_csv)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pair_audit = build_pair_status_audit(
        decoder,
        min_units=min_units,
        max_posterior_mean_error_cm_median=max_posterior_mean_error_cm_median,
        max_map_error_cm_median=max_map_error_cm_median,
        min_posterior_coverage_fraction=min_posterior_coverage_fraction,
    )
    failure_summary = summarize_failure_reasons(pair_audit)
    threshold_sensitivity = build_threshold_sensitivity(
        decoder,
        min_units_grid=min_units_grid,
        posterior_error_grid=posterior_error_grid,
        map_error_grid=map_error_grid,
        coverage_grid=coverage_grid,
    )
    metric_distribution = build_metric_distribution(decoder, units)
    gates = build_gate_summary(
        decoder=decoder,
        units=units,
        pairs=pairs,
        pair_audit=pair_audit,
        failure_summary=failure_summary,
        threshold_sensitivity=threshold_sensitivity,
        metric_distribution=metric_distribution,
        expected_threshold_rows=len(tuple(min_units_grid))
        * len(tuple(posterior_error_grid))
        * len(tuple(map_error_grid))
        * len(tuple(coverage_grid)),
    )

    pair_audit.to_csv(out / PAIR_AUDIT_OUTPUT, index=False)
    failure_summary.to_csv(out / FAILURE_OUTPUT, index=False)
    threshold_sensitivity.to_csv(out / SENSITIVITY_OUTPUT, index=False)
    metric_distribution.to_csv(out / METRIC_OUTPUT, index=False)
    gates.to_csv(out / GATE_OUTPUT, index=False)
    (out / SUMMARY_OUTPUT).write_text(
        build_markdown_summary(
            pair_audit,
            failure_summary,
            threshold_sensitivity,
            gates,
            min_units=min_units,
            max_posterior_mean_error_cm_median=max_posterior_mean_error_cm_median,
            max_map_error_cm_median=max_map_error_cm_median,
            min_posterior_coverage_fraction=min_posterior_coverage_fraction,
            overlap_metric_present="track_spike_position_overlap_s" in decoder.columns,
        ),
        encoding="utf-8",
    )
    return {
        "pair_audit": pair_audit,
        "failure_summary": failure_summary,
        "threshold_sensitivity": threshold_sensitivity,
        "metric_distribution": metric_distribution,
        "gates": gates,
    }


def load_decoder_qc(path: str | Path) -> pd.DataFrame:
    decoder = pd.read_csv(path)
    missing = sorted(REQUIRED_DECODER_COLUMNS.difference(decoder.columns))
    if missing:
        raise ValueError(f"decoder QC CSV is missing required columns: {missing}")
    prepared = decoder.copy()
    prepared["animal"] = prepared["animal"].astype(str).str.upper()
    prepared["date"] = prepared["date"].astype(str)
    prepared["track1_session"] = prepared["track1_session"].astype(str)
    prepared["sleeppost_session"] = prepared["sleeppost_session"].astype(str)
    return prepared


def load_unit_qc(path: str | Path) -> pd.DataFrame:
    units = pd.read_csv(path)
    missing = sorted(REQUIRED_UNIT_COLUMNS.difference(units.columns))
    if missing:
        raise ValueError(f"unit QC CSV is missing required columns: {missing}")
    prepared = units.copy()
    prepared["animal"] = prepared["animal"].astype(str).str.upper()
    prepared["date"] = prepared["date"].astype(str)
    prepared["track1_session"] = prepared["track1_session"].astype(str)
    return prepared


def load_pairs(path: str | Path) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    rename: dict[str, str] = {}
    if "track1_session" not in pairs.columns and "track_session" in pairs.columns:
        rename["track_session"] = "track1_session"
    if "sleeppost_session" not in pairs.columns and "sleepPOST_session" in pairs.columns:
        rename["sleepPOST_session"] = "sleeppost_session"
    pairs = pairs.rename(columns=rename)
    missing = sorted(REQUIRED_PAIR_COLUMNS.difference(pairs.columns))
    if missing:
        raise ValueError(f"pairs CSV is missing required columns: {missing}")
    prepared = pairs.copy()
    prepared["animal"] = prepared["animal"].astype(str).str.upper()
    prepared["date"] = prepared["date"].astype(str)
    prepared["track1_session"] = prepared["track1_session"].astype(str)
    prepared["sleeppost_session"] = prepared["sleeppost_session"].astype(str)
    return prepared


def build_pair_status_audit(
    decoder: pd.DataFrame,
    *,
    min_units: int,
    max_posterior_mean_error_cm_median: float,
    max_map_error_cm_median: float,
    min_posterior_coverage_fraction: float,
) -> pd.DataFrame:
    audit = decoder.copy()
    units = _numeric(audit, "encoding_units_passing_qc")
    posterior = _numeric(audit, "posterior_mean_error_cm_median")
    map_error = _numeric(audit, "map_error_cm_median")
    coverage = _numeric(audit, "posterior_coverage_fraction")
    if "track_spike_position_overlap_s" in audit.columns:
        overlap = _numeric(audit, "track_spike_position_overlap_s")
    else:
        overlap = pd.Series(np.nan, index=audit.index, dtype=float)
        audit["track_spike_position_overlap_s"] = np.nan

    finite_required = units.notna() & posterior.notna() & map_error.notna() & coverage.notna()
    failed_missing_overlap = overlap.notna() & (overlap <= 0.0)
    threshold_expected_passed = (
        finite_required
        & (units >= int(min_units))
        & (posterior <= float(max_posterior_mean_error_cm_median))
        & (map_error <= float(max_map_error_cm_median))
        & (coverage >= float(min_posterior_coverage_fraction))
        & ~failed_missing_overlap
    )
    decoder_passed = _pass_status_mask(audit["decoder_status"])

    audit["decoder_qc_status"] = audit["decoder_status"].astype(str)
    audit["decoder_qc_passed"] = decoder_passed
    audit["encoding_units_passing_qc"] = units
    audit["posterior_mean_error_cm_median"] = posterior
    audit["map_error_cm_median"] = map_error
    audit["posterior_coverage_fraction"] = coverage
    audit["track_spike_position_overlap_s"] = overlap
    audit["failed_min_units"] = units.notna() & (units < int(min_units))
    audit["failed_posterior_mean_error"] = posterior.notna() & (
        posterior > float(max_posterior_mean_error_cm_median)
    )
    audit["failed_map_error"] = map_error.notna() & (map_error > float(max_map_error_cm_median))
    audit["failed_posterior_coverage"] = coverage.notna() & (
        coverage < float(min_posterior_coverage_fraction)
    )
    audit["failed_missing_spike_position_overlap"] = failed_missing_overlap
    audit["failed_missing_metrics"] = ~finite_required
    audit["threshold_expected_passed"] = threshold_expected_passed
    audit["failed_schema"] = decoder_passed.ne(threshold_expected_passed)
    audit["primary_failure_reason"] = [
        primary_failure_reason(row)
        for row in audit[
            [
                "failed_missing_metrics",
                "failed_missing_spike_position_overlap",
                "failed_min_units",
                "failed_posterior_mean_error",
                "failed_map_error",
                "failed_posterior_coverage",
                "failed_schema",
            ]
        ].itertuples(index=False)
    ]
    return audit.reindex(columns=PAIR_AUDIT_COLUMNS)


def primary_failure_reason(row: tuple[object, ...]) -> str:
    (
        failed_missing_metrics,
        failed_missing_spike_position_overlap,
        failed_min_units,
        failed_posterior_mean_error,
        failed_map_error,
        failed_posterior_coverage,
        failed_schema,
    ) = [bool(value) for value in row]
    if failed_missing_metrics:
        return "missing_finite_decoder_metrics"
    if failed_missing_spike_position_overlap:
        return "missing_spike_position_overlap"
    if failed_min_units:
        return "too_few_encoding_units"
    if failed_posterior_mean_error:
        return "poor_posterior_mean_error"
    if failed_map_error:
        return "poor_map_error"
    if failed_posterior_coverage:
        return "low_posterior_coverage"
    if failed_schema:
        return "schema_status_mismatch"
    return "pass"


def summarize_failure_reasons(pair_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = int(len(pair_audit))
    for reason, column in FAILURE_REASONS:
        failed = pair_audit[column].map(_as_bool) if column in pair_audit.columns else pd.Series(False, index=pair_audit.index)
        affected = pair_audit[failed]
        rows.append(
            {
                "failure_reason": reason,
                "failed_pairs": int(failed.sum()),
                "total_pairs": total,
                "fraction_pairs": float(failed.mean()) if total else 0.0,
                "animals_affected": int(affected["animal"].astype(str).nunique()) if not affected.empty else 0,
                "pairs_affected": ";".join(_pair_ids(affected)),
            }
        )
    primary_counts = pair_audit["primary_failure_reason"].value_counts(dropna=False).to_dict()
    for row in rows:
        row["primary_failure_pairs"] = int(primary_counts.get(row["failure_reason"], 0))
    return pd.DataFrame(
        rows,
        columns=[
            "failure_reason",
            "failed_pairs",
            "total_pairs",
            "fraction_pairs",
            "animals_affected",
            "primary_failure_pairs",
            "pairs_affected",
        ],
    )


def build_threshold_sensitivity(
    decoder: pd.DataFrame,
    *,
    min_units_grid: Sequence[int],
    posterior_error_grid: Sequence[float],
    map_error_grid: Sequence[float],
    coverage_grid: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    units = _numeric(decoder, "encoding_units_passing_qc")
    posterior = _numeric(decoder, "posterior_mean_error_cm_median")
    map_error = _numeric(decoder, "map_error_cm_median")
    coverage = _numeric(decoder, "posterior_coverage_fraction")
    finite = units.notna() & posterior.notna() & map_error.notna() & coverage.notna()
    for idx, (min_units, max_posterior, max_map, min_coverage) in enumerate(
        product(min_units_grid, posterior_error_grid, map_error_grid, coverage_grid),
        start=1,
    ):
        keep = (
            finite
            & (units >= int(min_units))
            & (posterior <= float(max_posterior))
            & (map_error <= float(max_map))
            & (coverage >= float(min_coverage))
        )
        retained = decoder[keep].copy()
        rows.append(
            {
                "threshold_set_id": f"threshold_{idx:03d}",
                "min_units": int(min_units),
                "max_posterior_mean_error_cm_median": float(max_posterior),
                "max_map_error_cm_median": float(max_map),
                "min_posterior_coverage_fraction": float(min_coverage),
                "decoder_pass_pairs": int(keep.sum()),
                "animals_retained": int(retained["animal"].astype(str).str.upper().nunique()) if not retained.empty else 0,
                "pairs_retained": ";".join(_pair_ids(retained)),
                "median_posterior_mean_error_cm": _median(retained, "posterior_mean_error_cm_median"),
                "median_map_error_cm": _median(retained, "map_error_cm_median"),
                "median_posterior_coverage_fraction": _median(retained, "posterior_coverage_fraction"),
            }
        )
    return pd.DataFrame(rows)


def build_metric_distribution(decoder: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    decoder_metrics = [
        "n_units_track1",
        "n_position_samples",
        "track_duration_s",
        "valid_position_fraction",
        "linearized_track_span_cm",
        "occupancy_nonzero_bins",
        "encoding_units_passing_qc",
        "median_firing_rate_hz",
        "n_place_like_units",
        "crossval_n_folds",
        "posterior_mean_error_cm_median",
        "posterior_mean_error_cm_p75",
        "posterior_mean_error_cm_p90",
        "map_error_cm_median",
        "map_error_cm_p75",
        "map_error_cm_p90",
        "posterior_coverage_fraction",
        "track_spike_position_overlap_s",
    ]
    unit_metrics = [
        "n_spikes_track1",
        "mean_rate_hz",
        "peak_rate_hz",
        "spatial_information",
        "place_field_peak_cm",
        "place_field_width_cm",
        "occupancy_covered_fraction",
    ]
    for metric in decoder_metrics:
        if metric in decoder.columns:
            rows.append(metric_distribution_row("decoder", metric, decoder[metric]))
    for metric in unit_metrics:
        if metric in units.columns:
            rows.append(metric_distribution_row("unit", metric, units[metric]))
    return pd.DataFrame(
        rows,
        columns=[
            "scope",
            "metric",
            "count",
            "finite_count",
            "missing_count",
            "min",
            "p25",
            "median",
            "p75",
            "max",
            "mean",
        ],
    )


def metric_distribution_row(scope: str, metric: str, values: pd.Series) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return {
        "scope": scope,
        "metric": metric,
        "count": int(len(values)),
        "finite_count": int(len(finite)),
        "missing_count": int(len(values) - len(finite)),
        "min": _quantile(finite, 0.0),
        "p25": _quantile(finite, 0.25),
        "median": _quantile(finite, 0.50),
        "p75": _quantile(finite, 0.75),
        "max": _quantile(finite, 1.0),
        "mean": float(finite.mean()) if len(finite) else np.nan,
    }


def build_gate_summary(
    *,
    decoder: pd.DataFrame,
    units: pd.DataFrame,
    pairs: pd.DataFrame,
    pair_audit: pd.DataFrame,
    failure_summary: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    metric_distribution: pd.DataFrame,
    expected_threshold_rows: int,
) -> pd.DataFrame:
    usable_pairs = pairs[pairs["usable_pair"].map(_as_bool)] if "usable_pair" in pairs.columns else pairs
    decoder_ids = set(_pair_ids(decoder))
    expected_ids = set(_pair_ids(usable_pairs))
    nonempty_sensitivity = not threshold_sensitivity.empty
    max_pass_pairs = int(threshold_sensitivity["decoder_pass_pairs"].max()) if nonempty_sensitivity else 0
    gates = [
        _gate("decoder_qc_loaded", len(decoder) > 0, f"rows={len(decoder)}"),
        _gate("unit_qc_loaded", len(units) > 0, f"rows={len(units)}"),
        _gate("pairs_csv_loaded", len(pairs) > 0, f"rows={len(pairs)}; usable_pairs={len(usable_pairs)}"),
        _gate(
            "decoder_pair_coverage_complete",
            bool(expected_ids) and expected_ids.issubset(decoder_ids),
            f"decoder_pairs={len(decoder_ids)}; expected_usable_pairs={len(expected_ids)}",
        ),
        _gate(
            "failure_reason_summary_populated",
            not failure_summary.empty and failure_summary["failure_reason"].notna().all(),
            f"rows={len(failure_summary)}",
        ),
        _gate(
            "threshold_sensitivity_grid_complete",
            len(threshold_sensitivity) == int(expected_threshold_rows),
            f"rows={len(threshold_sensitivity)}; expected={expected_threshold_rows}",
        ),
        _gate(
            "metric_distribution_written",
            not metric_distribution.empty,
            f"rows={len(metric_distribution)}",
        ),
        _gate(
            "status_consistent_with_default_thresholds",
            int(pair_audit["failed_schema"].map(_as_bool).sum()) == 0,
            f"schema_status_mismatch_pairs={int(pair_audit['failed_schema'].map(_as_bool).sum())}",
        ),
        _gate(
            "threshold_sensitivity_retains_any_pair",
            max_pass_pairs > 0,
            f"max_decoder_pass_pairs={max_pass_pairs}",
        ),
    ]
    infrastructure_pass = all(bool(gate["passed"]) for gate in gates)
    gates.append(_gate("overall", infrastructure_pass, f"passed={sum(bool(g['passed']) for g in gates)}/{len(gates)}"))
    return pd.DataFrame(gates)


def build_markdown_summary(
    pair_audit: pd.DataFrame,
    failure_summary: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    gates: pd.DataFrame,
    *,
    min_units: int,
    max_posterior_mean_error_cm_median: float,
    max_map_error_cm_median: float,
    min_posterior_coverage_fraction: float,
    overlap_metric_present: bool,
) -> str:
    total_pairs = int(len(pair_audit))
    recorded_pass = int(pair_audit["decoder_qc_passed"].map(_as_bool).sum()) if total_pairs else 0
    expected_pass = int(pair_audit["threshold_expected_passed"].map(_as_bool).sum()) if total_pairs else 0
    best = (
        threshold_sensitivity.sort_values(["decoder_pass_pairs", "animals_retained"], ascending=False).head(8)
        if not threshold_sensitivity.empty
        else threshold_sensitivity
    )
    max_pass = int(threshold_sensitivity["decoder_pass_pairs"].max()) if not threshold_sensitivity.empty else 0
    max_animals = int(threshold_sensitivity["animals_retained"].max()) if not threshold_sensitivity.empty else 0
    if not pair_audit.empty:
        primary_counts = pair_audit["primary_failure_reason"].value_counts()
        primary = pd.DataFrame({"reason": primary_counts.index.astype(str), "pairs": primary_counts.to_numpy(dtype=int)})
    else:
        primary = pd.DataFrame(columns=["reason", "pairs"])
    lines = [
        "# Olafsdottir Track1 Decoder QC Triage",
        "",
        "This is a diagnostic-only report. It does not change decoder thresholds, pilot-event selection, or replay-evidence scoring.",
        "",
        "## Default Thresholds",
        "",
        _markdown_table(
            ["Parameter", "Value"],
            [
                ("minimum encoding units", min_units),
                ("posterior mean median error max cm", max_posterior_mean_error_cm_median),
                ("MAP median error max cm", max_map_error_cm_median),
                ("posterior coverage minimum", min_posterior_coverage_fraction),
            ],
        ),
        "",
        "## Overview",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ("pairs in decoder QC", total_pairs),
                ("recorded decoder pass pairs", recorded_pass),
                ("threshold-derived pass pairs", expected_pass),
                ("maximum pairs retained by sensitivity grid", max_pass),
                ("maximum animals retained by sensitivity grid", max_animals),
                ("spike/position overlap metric present", overlap_metric_present),
            ],
        ),
        "",
        "## Primary Failure Reasons",
        "",
        _markdown_table(["Reason", "Pairs"], primary.itertuples(index=False, name=None)),
        "",
        "## Failure Reason Summary",
        "",
        _markdown_table(
            ["Failure reason", "Failed pairs", "Animals affected"],
            failure_summary[["failure_reason", "failed_pairs", "animals_affected"]].itertuples(index=False, name=None),
        ),
        "",
        "## Most Permissive Passing Threshold Sets",
        "",
        _markdown_table(
            [
                "Threshold",
                "Min units",
                "Posterior max",
                "MAP max",
                "Coverage min",
                "Pairs",
                "Animals",
            ],
            best[
                [
                    "threshold_set_id",
                    "min_units",
                    "max_posterior_mean_error_cm_median",
                    "max_map_error_cm_median",
                    "min_posterior_coverage_fraction",
                    "decoder_pass_pairs",
                    "animals_retained",
                ]
            ].itertuples(index=False, name=None),
        ),
        "",
        "## Gate Summary",
        "",
        _markdown_table(["Gate", "Status", "Value"], gates[["gate", "status", "value"]].itertuples(index=False, name=None)),
        "",
    ]
    if not overlap_metric_present:
        lines.extend(
            [
                "## Notes",
                "",
                "`track_spike_position_overlap_s` was not present in the decoder QC input, so overlap failures could not be diagnosed from this triage input.",
                "",
            ]
        )
    return "\n".join(lines)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _pass_status_mask(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().eq("pass").fillna(False)


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "pass", "passed"}


def _pair_ids(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    required = ["animal", "date", "track1_session", "sleeppost_session"]
    if any(column not in frame.columns for column in required):
        return []
    rows = frame[required].fillna("").astype(str)
    return [
        f"{row.animal}|{row.date}|{row.track1_session}|{row.sleeppost_session}"
        for row in rows.itertuples(index=False)
    ]


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else float("nan")


def _quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.quantile(q)) if len(values) else np.nan


def _gate(gate: str, passed: bool, value: object) -> dict[str, object]:
    return {
        "gate": gate,
        "status": "pass" if bool(passed) else "fail",
        "passed": bool(passed),
        "value": value,
    }


def _markdown_table(headers: Sequence[object], rows: Iterable[Sequence[object]]) -> str:
    header_cells = [str(value) for value in headers]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join("---" for _ in header_cells) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoder-qc",
        type=Path,
        default=Path("results/olafsdottir-track1-decoder-qc/olafsdottir_track1_decoder_crossval_qc.csv"),
    )
    parser.add_argument(
        "--unit-qc",
        type=Path,
        default=Path("results/olafsdottir-track1-decoder-qc/olafsdottir_track1_encoding_unit_qc.csv"),
    )
    parser.add_argument("--pairs-csv", type=Path, default=Path("results/olafsdottir_track_sleep_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/olafsdottir-track1-decoder-qc-triage"))
    parser.add_argument("--min-units", type=int, default=DEFAULT_MIN_UNITS)
    parser.add_argument("--max-posterior-mean-error-cm-median", type=float, default=DEFAULT_MAX_POSTERIOR_ERROR_CM)
    parser.add_argument("--max-map-error-cm-median", type=float, default=DEFAULT_MAX_MAP_ERROR_CM)
    parser.add_argument("--min-posterior-coverage-fraction", type=float, default=DEFAULT_MIN_POSTERIOR_COVERAGE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tables = run_decoder_qc_triage(
        decoder_qc=args.decoder_qc,
        unit_qc=args.unit_qc,
        pairs_csv=args.pairs_csv,
        output_dir=args.output_dir,
        min_units=args.min_units,
        max_posterior_mean_error_cm_median=args.max_posterior_mean_error_cm_median,
        max_map_error_cm_median=args.max_map_error_cm_median,
        min_posterior_coverage_fraction=args.min_posterior_coverage_fraction,
    )
    print(tables["failure_summary"].to_string(index=False))
    print()
    print(tables["gates"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
