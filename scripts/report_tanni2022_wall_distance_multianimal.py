#!/usr/bin/env python3
"""Combine completed Tanni wall-distance runs without rescoring events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import build_script_provenance  # noqa: E402
from analyze_tanni2022_wall_distance_replay import (  # noqa: E402
    association_summary,
    build_gate_summary,
    make_figure,
    wall_quartile_summary,
    write_summary,
)


TABLES = (
    "tanni2022_session_manifest.csv",
    "tanni2022_unit_qc.csv",
    "tanni2022_decoder_qc_samples.csv",
    "tanni2022_decoder_qc_summary.csv",
    "tanni2022_ripple_candidates.csv",
    "tanni2022_replay_speed_events.csv",
    "tanni2022_replay_speed_segments.csv",
    "tanni2022_synthetic_constant_speed_null.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20220714)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, pd.DataFrame] = {}
    input_paths: dict[str, Path] = {}
    for table_name in TABLES:
        frames = []
        for index, input_dir in enumerate(args.input_dir):
            path = input_dir.resolve() / table_name
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path))
            input_paths[f"input_{index}_{table_name}"] = path
        combined[table_name] = pd.concat(frames, ignore_index=True)
        combined[table_name].to_csv(output_dir / table_name, index=False)
    manifest = combined["tanni2022_session_manifest.csv"]
    unit_qc = combined["tanni2022_unit_qc.csv"]
    decoder_samples = combined["tanni2022_decoder_qc_samples.csv"]
    decoder_summary = combined["tanni2022_decoder_qc_summary.csv"]
    ripple_events = combined["tanni2022_ripple_candidates.csv"]
    event_speed = combined["tanni2022_replay_speed_events.csv"]
    segments = combined["tanni2022_replay_speed_segments.csv"]
    synthetic = combined["tanni2022_synthetic_constant_speed_null.csv"]
    quartiles = wall_quartile_summary(segments, decoder_samples)
    associations = association_summary(
        segments,
        synthetic,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    gates, verdict = build_gate_summary(
        manifest,
        unit_qc,
        decoder_summary,
        ripple_events,
        event_speed,
        segments,
        associations,
    )
    quartiles.to_csv(output_dir / "tanni2022_wall_distance_quartiles.csv", index=False)
    associations.to_csv(output_dir / "tanni2022_wall_distance_associations.csv", index=False)
    gates.to_csv(output_dir / "tanni2022_wall_distance_gate_summary.csv", index=False)
    make_figure(quartiles, associations, output_dir / "tanni2022_wall_distance_replay_figure.png")
    write_summary(
        output_dir / "tanni2022_wall_distance_replay_summary.md",
        manifest,
        unit_qc,
        decoder_summary,
        ripple_events,
        event_speed,
        associations,
        verdict,
    )
    provenance = build_script_provenance(input_paths=input_paths, argv=sys.argv)
    payload = {
        "dataset": "tanni_de_cothi_barry_2022_large_2d",
        "analysis": "non_rescoring_multianimal_wall_distance_report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directories": [str(path.resolve()) for path in args.input_dir],
        "animals": sorted(manifest["animal"].astype(str).unique().tolist()),
        "sessions": int(manifest.shape[0]),
        "verdict": verdict,
        "provenance": provenance,
    }
    (output_dir / "tanni2022_wall_distance_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
