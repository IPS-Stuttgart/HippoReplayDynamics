#!/usr/bin/env python3
"""Create replay-window variants for event-window sensitivity analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import event_window_variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, help="CSV with event_index,start,end columns")
    parser.add_argument("--output", default="results/event-window-sensitivity-plan.csv")
    parser.add_argument("--paddings-s", default="0,0.01,0.02")
    parser.add_argument("--min-duration-s", type=float, default=0.003)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    paddings = [float(item.strip()) for item in args.paddings_s.split(",") if item.strip()]
    plan = event_window_variants(events, paddings_s=paddings, min_duration_s=args.min_duration_s)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output, index=False)
    print(plan.head(20).to_string(index=False))
    print(f"Wrote {len(plan)} window variants to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
