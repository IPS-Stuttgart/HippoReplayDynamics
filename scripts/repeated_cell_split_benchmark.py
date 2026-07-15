#!/usr/bin/env python3
"""Run held-out replay benchmarks over repeated cell-split seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hipporeplayimm.benchmarks import BenchmarkConfig, BenchmarkResult, run_open_field_benchmark
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig
from hipporeplayimm.position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
)


def _parse_models(value: str) -> tuple[str, ...]:
    """Parse a comma/whitespace model list without dropping empty entries."""

    raw = str(value)
    if not raw.strip():
        raise ValueError("--models must contain at least one model")
    comma_parts = raw.split(",")
    if any(not part.strip() for part in comma_parts):
        raise ValueError("--models must not contain empty comma-separated entries")
    models: list[str] = []
    for part in comma_parts:
        models.extend(part.split())
    if not models:
        raise ValueError("--models must contain at least one model")
    return tuple(models)


def _parse_seeds(value: str) -> tuple[int, ...]:
    """Parse comma-separated non-negative random seeds without dropping entries."""

    raw = str(value)
    if not raw.strip():
        raise ValueError("--random-seeds must contain at least one seed")
    parts = raw.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError("--random-seeds must not contain empty comma-separated entries")

    seeds: list[int] = []
    for part in parts:
        token = part.strip()
        try:
            seed = int(token)
        except ValueError:
            raise ValueError("--random-seeds entries must be integers") from None
        if seed < 0:
            raise ValueError("--random-seeds entries must be non-negative")
        seeds.append(seed)
    return tuple(seeds)


def _aggregate_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    numeric = [
        column
        for column in (
            "heldout_log_likelihood",
            "delta_vs_best_static",
            "bits_per_spike_vs_best_static",
            "lower_bound_delta_vs_best_static",
            "lower_bound_bits_per_spike_vs_best_static",
        )
        if column in rows
    ]
    if not numeric:
        return pd.DataFrame()
    grouped = rows.groupby("model")
    out = grouped[numeric].agg(["mean", "median", "std"]).reset_index()
    out.columns = ["_".join(str(part) for part in column if str(part)) for column in out.columns]
    if "split_seed" in rows:
        seed_counts = (
            rows.groupby("model", as_index=False)["split_seed"]
            .nunique()
            .rename(columns={"split_seed": "split_seeds"})
        )
        out = out.merge(seed_counts, on="model", how="left")
    sort_column = f"{numeric[0]}_mean"
    return out.sort_values(sort_column, ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run open-field held-out benchmarks across repeated cell-split seeds.")
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-seeds", default="1,2,3,4,5")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--models", default="random,stationary,diffusion,momentum,imm")
    parser.add_argument("--test-cell-fraction", type=float, default=0.25)
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument("--time-bin-ms", type=float, default=3.0)
    parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    parser.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    parser.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    args = parser.parse_args()
    try:
        random_seeds = _parse_seeds(args.random_seeds)
        models = _parse_models(args.models)
    except ValueError as exc:
        parser.error(str(exc))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    for seed in random_seeds:
        config = BenchmarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=args.bin_size_cm,
                smoothing_sigma_bins=args.smoothing_sigma_bins,
                min_speed_cm_s=args.min_speed_cm_s,
            ),
            emissions=EmissionConfig(
                time_bin_s=args.time_bin_ms / 1000.0,
                spike_rate_scale=args.spike_rate_scale,
            ),
            test_cell_fraction=args.test_cell_fraction,
            max_events_per_session=args.max_events,
            candidate_top_k=args.candidate_top_k,
            random_seed=seed,
            models=models,
        )
        result: BenchmarkResult = run_open_field_benchmark(args.root, config)
        rows = result.rows.copy()
        rows["split_seed"] = seed
        all_rows.append(rows)
        summary = result.summary().copy()
        summary["split_seed"] = seed
        all_summaries.append(summary)
        print(f"Finished split seed {seed}: {len(rows)} event/model rows")

    event_scores = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_by_seed = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    summary_across_seeds = _aggregate_summary(event_scores)
    event_scores.to_csv(output / "event_scores_repeated_splits.csv", index=False)
    summary_by_seed.to_csv(output / "summary_by_split_seed.csv", index=False)
    summary_across_seeds.to_csv(output / "summary_across_split_seeds.csv", index=False)
    print(summary_across_seeds.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
