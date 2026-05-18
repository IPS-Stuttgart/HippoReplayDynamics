#!/usr/bin/env python3
"""Aggregate event-sharded project model-evidence outputs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from benchmark_model_evidence import _add_evidence_columns, _counts, _summary, _write
from model_evidence_support_audit import write_evidence_support_audit

_CONSTANT_SETTING_COLUMNS = (
    "bin_size_cm",
    "smoothing_sigma_bins",
    "min_speed_cm_s",
    "time_bin_s",
    "spike_rate_scale",
    "clusterless_mark_smoothing_sigma_bins",
    "clusterless_mark_prior_count",
    "clusterless_mark_variance_floor",
    "clusterless_rate_floor_hz",
)


def _load_score_files(shard_glob: str) -> list[Path]:
    paths = sorted(Path(path) for path in glob.glob(shard_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No model-evidence shard CSVs matched: {shard_glob}")
    return paths


def _validate_constant_settings(combined: pd.DataFrame) -> None:
    """Reject aggregates that silently mix incompatible benchmark settings."""

    inconsistent: dict[str, list[str]] = {}
    for column in _CONSTANT_SETTING_COLUMNS:
        if column not in combined.columns:
            continue
        values = combined[column].dropna().unique()
        if len(values) > 1:
            inconsistent[column] = sorted(str(value) for value in values)

    if not inconsistent:
        return

    lines = ["Model-evidence shards mix incompatible run settings:"]
    for column, values in sorted(inconsistent.items()):
        lines.append(f"- {column}: {', '.join(values)}")
    raise ValueError("\n".join(lines))


def aggregate(shard_glob: str, outdir: Path) -> pd.DataFrame:
    frames = []
    for path in _load_score_files(shard_glob):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_shard_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise RuntimeError("All model-evidence shard CSVs were empty.")

    combined = pd.concat(frames, ignore_index=True)
    duplicate_key = ["session", "event_index", "model"]
    duplicates = combined.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        duplicate_rows = combined.loc[duplicates, duplicate_key + ["source_shard_file"]]
        raise ValueError(
            "Model-evidence shards contain duplicate event/model rows:\n"
            + duplicate_rows.head(20).to_string(index=False)
        )
    _validate_constant_settings(combined)

    combined = _add_evidence_columns(combined.drop(columns=["source_shard_file"]))
    _write(combined, outdir)
    write_evidence_support_audit(combined, outdir)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate event-sharded project model-evidence outputs.")
    parser.add_argument("--shard-glob", required=True)
    parser.add_argument("--output", default="results/model-evidence")
    args = parser.parse_args()

    combined = aggregate(args.shard_glob, Path(args.output))
    print(_summary(combined).to_string(index=False))
    print("\nBest-model counts:")
    print(_counts(combined).to_string(index=False))
    print(f"\nRows: {len(combined)}")
    if "status" in combined:
        print(f"Failures: {int((combined['status'] != 'success').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
