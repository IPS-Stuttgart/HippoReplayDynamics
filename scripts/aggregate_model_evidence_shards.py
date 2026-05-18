#!/usr/bin/env python3
"""Aggregate event-sharded project model-evidence outputs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from benchmark_model_evidence import _add_evidence_columns, _counts, _summary, _write
from model_evidence_settings import _validate_constant_settings
from model_evidence_support_audit import write_evidence_support_audit


def _load_score_files(shard_glob: str) -> list[Path]:
    paths = sorted(Path(path) for path in glob.glob(shard_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No model-evidence shard CSVs matched: {shard_glob}")
    return paths


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
