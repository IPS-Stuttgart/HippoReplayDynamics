#!/usr/bin/env python3
"""Plan event shards for KD-aligned model-evidence workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_kd_model_evidence import _check_session, _events, _session_path
from event_shard_count_validation import optional_nonnegative_integer_count, positive_integer_count
from hipporeplayimm.data import load_replay_session


def _event_spec(event_ids: list[int]) -> str:
    if not event_ids:
        raise ValueError("Cannot build an event spec for an empty event shard.")
    ranges: list[str] = []
    start = event_ids[0]
    prev = event_ids[0]
    for event_id in event_ids[1:]:
        if event_id == prev + 1:
            prev = event_id
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = event_id
        prev = event_id
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _event_chunks(event_ids: list[int], requested_shards: int) -> list[list[int]]:
    validated_requested_shards = positive_integer_count("--event-shard-count", requested_shards)
    shard_count = min(validated_requested_shards, len(event_ids))
    if shard_count < 1:
        raise ValueError("No events selected.")
    chunks: list[list[int]] = []
    for shard_index in range(shard_count):
        start = shard_index * len(event_ids) // shard_count
        stop = (shard_index + 1) * len(event_ids) // shard_count
        chunks.append(event_ids[start:stop])
    return chunks


def _validate_max_events(max_events: int | None) -> int | None:
    return optional_nonnegative_integer_count("--max-events", max_events)


def plan_event_shards(
    dataset_root: Path,
    session_id: str,
    events: str,
    *,
    max_events: int | None,
    event_shard_count: int,
    momentum_shard_count: int,
) -> dict[str, object]:
    validated_max_events = _validate_max_events(max_events)
    validated_event_shard_count = positive_integer_count("--event-shard-count", event_shard_count)
    validated_momentum_shard_count = positive_integer_count("--momentum-shard-count", momentum_shard_count)
    session_dir = _session_path(dataset_root, session_id)
    _check_session(session_dir)
    session = load_replay_session(session_dir)
    event_ids = _events(events, session, validated_max_events)
    chunks = _event_chunks(event_ids, validated_event_shard_count)
    matrix_size = len(chunks) * validated_momentum_shard_count
    if matrix_size > 256:
        raise ValueError(
            f"Requested event/grid shard matrix has {matrix_size} jobs, exceeding GitHub Actions' 256-job matrix limit. "
            "Reduce --event-shard-count or --momentum-shard-count."
        )
    event_matrix = [
        {
            "event_shard_index": shard_index,
            "events_spec": _event_spec(chunk),
            "event_count": len(chunk),
            "first_event_index": int(chunk[0]),
            "last_event_index": int(chunk[-1]),
        }
        for shard_index, chunk in enumerate(chunks)
    ]
    return {
        "session": session.session_id,
        "events": events,
        "max_events": validated_max_events,
        "event_count": len(event_ids),
        "requested_event_shard_count": validated_event_shard_count,
        "event_shard_count": len(event_matrix),
        "momentum_shard_count": validated_momentum_shard_count,
        "grid_shard_indices": list(range(validated_momentum_shard_count)),
        "event_matrix": event_matrix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan event shards for KD-aligned model-evidence workflows.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="run")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--event-shard-count", type=int, default=8)
    parser.add_argument("--momentum-shard-count", type=int, default=10)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    plan = plan_event_shards(
        Path(args.dataset_root),
        args.session,
        args.events,
        max_events=args.max_events,
        event_shard_count=args.event_shard_count,
        momentum_shard_count=args.momentum_shard_count,
    )
    text = json.dumps(plan, indent=2)
    if args.output:
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
