"""Command-line interface for HippoReplayIMM."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmarks import BenchmarkConfig, bootstrap_delta_ci, run_open_field_benchmark
from .data import load_open_field_sessions
from .encoding import build_emissions, fit_place_field_encoding
from .models import CandidateKinematicModel, RandomModel, StationaryModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hipporeplayimm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("root")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("root")
    benchmark_parser.add_argument("--preset", default="open-field-loso")
    benchmark_parser.add_argument("--output")
    benchmark_parser.add_argument("--max-events", type=int)
    benchmark_parser.add_argument("--candidate-top-k", type=int, default=64)

    decode_parser = subparsers.add_parser("decode-event")
    decode_parser.add_argument("root")
    decode_parser.add_argument("--session", required=True)
    decode_parser.add_argument("--event-id", type=int, required=True)
    decode_parser.add_argument("--candidate-top-k", type=int, default=64)

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.root)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "decode-event":
        return _decode_event(args)
    raise ValueError(args.command)


def _inspect(root: str) -> int:
    sessions = load_open_field_sessions(root)
    rows = [
        {
            "session": session.session_id,
            "position_frames": session.position.shape[0],
            "spikes": session.spikes.shape[0],
            "cells": session.cell_ids.shape[0],
            "excitatory_cells": session.excitatory_neurons.shape[0],
            "ripples": session.ripple_count,
            "run_ripples": session.ripple_indices_in_run().shape[0],
        }
        for session in sessions
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    if args.preset != "open-field-loso":
        raise ValueError("Only --preset open-field-loso is currently implemented")
    config = BenchmarkConfig(
        max_events_per_session=args.max_events,
        candidate_top_k=args.candidate_top_k,
    )
    result = run_open_field_benchmark(args.root, config)
    print(result.summary().to_string(index=False))
    if not result.rows.empty and "imm" in set(result.rows["model"]):
        ci_low, ci_high = bootstrap_delta_ci(result.rows, model="imm")
        print(f"IMM mean delta bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        result.rows.to_csv(output / "event_scores.csv", index=False)
        result.summary().to_csv(output / "summary.csv", index=False)
    return 0


def _decode_event(args: argparse.Namespace) -> int:
    sessions = {session.session_id: session for session in load_open_field_sessions(args.root)}
    if args.session not in sessions:
        available = ", ".join(sorted(sessions))
        raise KeyError(f"Unknown session {args.session!r}; available: {available}")
    session = sessions[args.session]
    encoding = fit_place_field_encoding(session)
    emissions = build_emissions(session, encoding, args.event_id)
    models = [
        RandomModel(),
        StationaryModel(),
        CandidateKinematicModel(mode="diffusion", top_k=args.candidate_top_k),
        CandidateKinematicModel(mode="momentum", top_k=args.candidate_top_k),
        CandidateKinematicModel(mode="imm", top_k=args.candidate_top_k),
    ]
    rows = []
    for model in models:
        score = model.score(emissions, encoding.bin_centers)
        rows.append(
            {
                "model": score.model_name,
                "log_likelihood": score.log_likelihood,
                "n_spikes": score.n_spikes,
                **score.diagnostics,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
