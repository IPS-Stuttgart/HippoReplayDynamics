"""Command-line interface for HippoReplayIMM."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmarks import BenchmarkConfig, bootstrap_delta_ci, run_open_field_benchmark
from .data import load_open_field_sessions
from .encoding import build_emissions, fit_place_field_encoding
from .ground_truth import (
    GroundTruthConfig,
    compare_scores_to_ground_truth,
    generate_behavioral_ground_truth,
)
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

    ground_truth_parser = subparsers.add_parser("ground-truth")
    ground_truth_parser.add_argument("root")
    ground_truth_parser.add_argument("--output", required=True)
    ground_truth_parser.add_argument("--max-events", type=int)
    ground_truth_parser.add_argument("--visit-radius-cm", type=float, default=10.0)
    ground_truth_parser.add_argument("--min-dwell-s", type=float, default=0.2)
    ground_truth_parser.add_argument("--future-horizon-s", type=float, default=30.0)

    compare_parser = subparsers.add_parser("compare-ground-truth")
    compare_parser.add_argument("root")
    compare_parser.add_argument("--scores", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--ground-truth")
    compare_parser.add_argument("--candidate-top-k", type=int, default=64)
    compare_parser.add_argument("--visit-radius-cm", type=float, default=10.0)
    compare_parser.add_argument("--min-dwell-s", type=float, default=0.2)
    compare_parser.add_argument("--future-horizon-s", type=float, default=30.0)

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.root)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "decode-event":
        return _decode_event(args)
    if args.command == "ground-truth":
        return _ground_truth(args)
    if args.command == "compare-ground-truth":
        return _compare_ground_truth(args)
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


def _ground_truth(args: argparse.Namespace) -> int:
    config = _ground_truth_config_from_args(args)
    frame = generate_behavioral_ground_truth(
        args.root,
        config=config,
        max_events_per_session=args.max_events,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(_ground_truth_summary(frame).to_string(index=False))
    return 0


def _compare_ground_truth(args: argparse.Namespace) -> int:
    config = _ground_truth_config_from_args(args)
    frame = compare_scores_to_ground_truth(
        args.root,
        args.scores,
        ground_truth=args.ground_truth,
        ground_truth_config=config,
        candidate_top_k=args.candidate_top_k,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(_comparison_summary(frame).to_string(index=False))
    return 0


def _ground_truth_config_from_args(args: argparse.Namespace) -> GroundTruthConfig:
    return GroundTruthConfig(
        visit_radius_cm=args.visit_radius_cm,
        min_dwell_s=args.min_dwell_s,
        future_horizon_s=args.future_horizon_s,
    )


def _ground_truth_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("session", as_index=False)
        .agg(
            events=("event_index", "count"),
            valid_labels=("valid_label", "sum"),
            median_time_to_arrival_s=("time_to_arrival_s", "median"),
        )
        .sort_values("session")
    )


def _comparison_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    valid = frame[frame["valid_label"].fillna(False)]
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby("model", as_index=False)
        .agg(
            rows=("event_index", "count"),
            goal_accuracy=("goal_correct", "mean"),
            median_endpoint_error_cm=("endpoint_error_cm", "median"),
            mean_true_well_posterior=("true_well_posterior", "mean"),
        )
        .sort_values("model")
    )


if __name__ == "__main__":
    raise SystemExit(main())
