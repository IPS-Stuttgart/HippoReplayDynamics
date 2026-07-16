#!/usr/bin/env python3
"""Merge hc-11 control shards and recompute all strict learning summaries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
import audit_hc11_pre_post_learning_controls as audit  # noqa: E402


def load_control_shards(shard_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    control_frames: list[pd.DataFrame] = []
    heldout_frames: list[pd.DataFrame] = []
    for shard_dir in shard_dirs:
        control_path = shard_dir / audit.CONTROL_EVIDENCE_OUTPUT
        heldout_path = shard_dir / audit.HELDOUT_OUTPUT
        if not control_path.is_file() or not heldout_path.is_file():
            raise FileNotFoundError(f"incomplete control shard: {shard_dir}")
        control_frames.append(pd.read_csv(control_path))
        heldout_frames.append(pd.read_csv(heldout_path))
    controls = pd.concat(control_frames, ignore_index=True)
    heldout = pd.concat(heldout_frames, ignore_index=True)
    control_key = [
        "session",
        "phase",
        "event_id",
        "population",
        "control_type",
        "replicate",
    ]
    heldout_key = ["session", "phase", "event_id", "population", "split_index"]
    if controls.duplicated(control_key).any():
        raise ValueError("control shards contain duplicate event/control replicates")
    if heldout.duplicated(heldout_key).any():
        raise ValueError("control shards contain duplicate held-out splits")
    return controls, heldout


def merge_and_summarize(
    shard_dirs: list[Path],
    *,
    margin_threshold: float,
    n_map_permutations: int,
    n_time_shuffles: int,
    n_heldout_splits: int,
    n_animal_bootstraps: int,
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    controls, heldout = load_control_shards(shard_dirs)
    successful_heldout = heldout[heldout["status"].eq("success")].copy()
    events = audit.build_event_summary(controls, successful_heldout, margin_threshold)
    contrasts = audit.learning_contrasts(events)
    populations = audit.summarize_populations(events, contrasts)
    session_effects, animal_effects = audit.learning_effects_by_session_and_animal(contrasts)
    inference, leave_one_out = audit.infer_equal_animal_learning_effects(
        animal_effects,
        n_bootstraps=n_animal_bootstraps,
        seed=random_seed,
    )
    gates = audit.gate_summary(
        controls,
        heldout,
        events,
        contrasts,
        n_map_permutations,
        n_time_shuffles,
        n_heldout_splits,
        inference,
    )
    return {
        audit.CONTROL_EVIDENCE_OUTPUT: controls,
        audit.HELDOUT_OUTPUT: heldout,
        audit.EVENT_OUTPUT: events,
        audit.CONTRAST_OUTPUT: contrasts,
        audit.POPULATION_OUTPUT: populations,
        audit.SESSION_EFFECT_OUTPUT: session_effects,
        audit.ANIMAL_EFFECT_OUTPUT: animal_effects,
        audit.INFERENCE_OUTPUT: inference,
        audit.LEAVE_ONE_ANIMAL_OUT_OUTPUT: leave_one_out,
        audit.GATE_OUTPUT: gates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--n-map-permutations", type=int, default=20)
    parser.add_argument("--n-time-shuffles", type=int, default=20)
    parser.add_argument("--n-heldout-splits", type=int, default=20)
    parser.add_argument("--n-animal-bootstraps", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    shard_dirs = [Path(value).resolve() for value in args.shard_dir]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = merge_and_summarize(
        shard_dirs,
        margin_threshold=args.margin_threshold,
        n_map_permutations=args.n_map_permutations,
        n_time_shuffles=args.n_time_shuffles,
        n_heldout_splits=args.n_heldout_splits,
        n_animal_bootstraps=args.n_animal_bootstraps,
        random_seed=args.random_seed,
    )
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "merged_hc11_pre_post_learning_control_shards",
        "shard_dirs": [str(path) for path in shard_dirs],
        "parameters": {
            "margin_threshold": args.margin_threshold,
            "n_map_permutations": args.n_map_permutations,
            "n_time_shuffles": args.n_time_shuffles,
            "n_heldout_splits": args.n_heldout_splits,
            "n_animal_bootstraps": args.n_animal_bootstraps,
            "random_seed": args.random_seed,
        },
        "outputs": list(outputs),
        **build_script_provenance(
            input_paths={
                f"shard_{index}_controls": path / audit.CONTROL_EVIDENCE_OUTPUT
                for index, path in enumerate(shard_dirs)
            }
            | {
                f"shard_{index}_heldout": path / audit.HELDOUT_OUTPUT
                for index, path in enumerate(shard_dirs)
            },
            cwd=ROOT,
            argv=sys.argv,
        ),
    }
    (output_dir / audit.MANIFEST_OUTPUT).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / audit.SUMMARY_OUTPUT).write_text(
        audit.build_markdown_summary(
            outputs[audit.POPULATION_OUTPUT],
            outputs[audit.INFERENCE_OUTPUT],
            outputs[audit.GATE_OUTPUT],
        )
    )
    print(
        f"Merged {len(shard_dirs)} shards and {len(outputs[audit.EVENT_OUTPUT])} "
        f"event-population rows into {output_dir}"
    )


if __name__ == "__main__":
    main()
