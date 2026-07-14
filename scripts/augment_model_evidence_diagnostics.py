#!/usr/bin/env python3
"""Add reliability, runtime, and parameter-provenance diagnostics to evidence CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reliability import add_event_reliability_flags


def _numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float)


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    d = _numeric_series(denom).replace(0.0, np.nan)
    return _numeric_series(num) / d


def _finite_quantile(values: pd.Series, q: float) -> float:
    numeric = _numeric_series(values).dropna()
    if numeric.empty:
        return float("nan")
    return float(np.quantile(numeric.to_numpy(dtype=float), q))


def augment(args: argparse.Namespace) -> pd.DataFrame:
    scores = pd.read_csv(args.scores)
    scores = add_event_reliability_flags(
        scores,
        min_spikes=args.min_spikes,
        min_time_bins=args.min_time_bins,
        min_candidate_log_mass=args.min_candidate_log_mass,
        max_terminal_entropy=args.max_terminal_entropy,
    )
    scores["hyperparameter_source"] = args.hyperparameter_source
    scores["selection_dataset"] = args.selection_dataset
    scores["selection_metric"] = args.selection_metric
    if "runtime_s" in scores:
        scores["runtime_s"] = _numeric_series(scores["runtime_s"])
        if "relative_log_evidence" in scores:
            scores["relative_log_evidence_per_runtime_s"] = _safe_divide(scores["relative_log_evidence"], scores["runtime_s"])
        if "truncated_relative_log_evidence" in scores:
            scores["truncated_relative_log_evidence_per_runtime_s"] = _safe_divide(scores["truncated_relative_log_evidence"], scores["runtime_s"])
    return scores


def write_outputs(scores: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "event_model_evidence_augmented.csv", index=False)
    if not scores.empty:
        reliability = scores.groupby("model", as_index=False, dropna=False).agg(
            rows=("model", "size"),
            reliable_rows=("event_reliable", "sum"),
            low_spike_rows=("event_low_spike_count", "sum"),
            low_candidate_mass_rows=("event_low_candidate_mass", "sum"),
            too_few_time_bin_rows=("event_too_few_time_bins", "sum"),
        )
        reliability["reliable_fraction"] = reliability["reliable_rows"] / reliability["rows"].clip(lower=1)
        reliability.to_csv(output / "model_reliability_summary.csv", index=False)
    if "runtime_s" in scores:
        runtime_scores = scores.copy()
        runtime_scores["runtime_s"] = _numeric_series(runtime_scores["runtime_s"])
        runtime = runtime_scores.groupby("model", as_index=False, dropna=False).agg(
            rows=("model", "size"),
            mean_runtime_s=("runtime_s", "mean"),
            median_runtime_s=("runtime_s", "median"),
            p95_runtime_s=("runtime_s", lambda x: _finite_quantile(x, 0.95)),
        )
        if "relative_log_evidence_per_runtime_s" in runtime_scores:
            runtime_scores["relative_log_evidence_per_runtime_s"] = _numeric_series(runtime_scores["relative_log_evidence_per_runtime_s"])
            runtime = runtime.merge(
                runtime_scores.groupby("model", as_index=False, dropna=False)["relative_log_evidence_per_runtime_s"].mean(),
                on="model",
                how="left",
            )
        runtime.to_csv(output / "model_runtime_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-spikes", type=int, default=3)
    parser.add_argument("--min-time-bins", type=int, default=2)
    parser.add_argument("--min-candidate-log-mass", type=float, default=float(np.log(0.95)))
    parser.add_argument("--max-terminal-entropy", type=float, default=float("inf"))
    parser.add_argument("--hyperparameter-source", default="unspecified")
    parser.add_argument("--selection-dataset", default="unspecified")
    parser.add_argument("--selection-metric", default="unspecified")
    args = parser.parse_args()
    scores = augment(args)
    write_outputs(scores, Path(args.output))
    print(scores.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
