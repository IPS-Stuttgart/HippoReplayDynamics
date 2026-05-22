#!/usr/bin/env python3
"""Generate strict-vs-certified simulation-recovery diagnostic tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hipporeplayimm.recovery_diagnostics import (
    build_recovery_diagnostic_tables,
    load_recovery_score_tables,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose synthetic replay-dynamics recovery failures by separating strict exact-comparable recovery, "
            "certified lower-bound recovery, and candidate-support coverage."
        )
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help="One or more simulation_recovery_event_scores.csv files or directories containing them.",
    )
    parser.add_argument("--output", required=True, help="Output directory for diagnostic tables and report.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scores = load_recovery_score_tables(args.scores)
    tables = build_recovery_diagnostic_tables(scores, source_paths=args.scores)
    tables.write(Path(args.output))
    print(tables.summary.to_string(index=False))
    print(f"\nWrote simulation-recovery diagnostics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
