"""Command-line interface for HippoReplayIMM."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmarks import BenchmarkConfig, _build_models, bootstrap_delta_ci, run_open_field_benchmark
from .data import load_open_field_sessions
from .encoding import build_emissions, fit_place_field_encoding
from .ground_truth import (
    GroundTruthConfig,
    compare_scores_to_ground_truth,
    generate_behavioral_ground_truth,
)
from .sweeps import (
    PyRecEstSweepConfig,
    pareto_sweep_summary,
    run_pyrecest_parameter_sweep,
    write_pyrecest_sweep_outputs,
)


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
    benchmark_parser.add_argument(
        "--models",
        default="random,stationary,diffusion,momentum,imm",
        help="Comma-separated model names to benchmark.",
    )
    _add_pyrecest_scalar_arguments(benchmark_parser)

    decode_parser = subparsers.add_parser("decode-event")
    decode_parser.add_argument("root")
    decode_parser.add_argument("--session", required=True)
    decode_parser.add_argument("--event-id", type=int, required=True)
    decode_parser.add_argument("--candidate-top-k", type=int, default=64)
    decode_parser.add_argument(
        "--models",
        default="random,stationary,diffusion,momentum,imm",
        help="Comma-separated model names to score.",
    )
    _add_pyrecest_scalar_arguments(decode_parser)

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
    _add_pyrecest_scalar_arguments(compare_parser)
    compare_parser.add_argument("--visit-radius-cm", type=float, default=10.0)
    compare_parser.add_argument("--min-dwell-s", type=float, default=0.2)
    compare_parser.add_argument("--future-horizon-s", type=float, default=30.0)

    sweep_parser = subparsers.add_parser("sweep-pyrecest")
    sweep_parser.add_argument("root")
    sweep_parser.add_argument("--output", required=True)
    sweep_parser.add_argument("--max-events", type=int, default=1)
    sweep_parser.add_argument("--candidate-top-k", type=int, default=64)
    sweep_parser.add_argument("--random-seed", type=int, default=1)
    sweep_parser.add_argument("--event-epoch", choices=("run", "all"), default="run")
    sweep_parser.add_argument(
        "--pyrecest-models",
        default="pyrecest-goal-particle",
        help="Comma-separated PyRecEst replay models to sweep.",
    )
    sweep_parser.add_argument(
        "--baseline-models",
        default="random,stationary",
        help=(
            "Comma-separated static/legacy models to include in each sweep run; "
            "use 'none' to skip."
        ),
    )
    sweep_parser.add_argument("--particles", default="128")
    sweep_parser.add_argument("--alpha", default="0.8")
    sweep_parser.add_argument("--beta", default="1.0")
    sweep_parser.add_argument("--process-noise-sigma-cm-s", default="60.0")
    sweep_parser.add_argument("--position-jump-sigma-cm", default="25.0")
    sweep_parser.add_argument("--jump-probability", default="0.03")
    sweep_parser.add_argument("--goal-reset-probability", default="0.02")
    sweep_parser.add_argument("--initial-velocity-sigma-cm-s", default="120.0")
    sweep_parser.add_argument("--imm-mode-stickiness", default="0.95")
    sweep_parser.add_argument("--imm-stationary-velocity-decay", default="0.0")
    sweep_parser.add_argument("--imm-diffusion-velocity-decay", default="0.0")
    sweep_parser.add_argument("--imm-momentum-velocity-decay", default="0.95")
    sweep_parser.add_argument("--imm-jump-fraction", default="0.9")
    sweep_parser.add_argument("--imm-jump-velocity-decay", default="0.25")
    sweep_parser.add_argument("--skip-ground-truth", action="store_true")
    sweep_parser.add_argument("--visit-radius-cm", type=float, default=10.0)
    sweep_parser.add_argument("--min-dwell-s", type=float, default=0.2)
    sweep_parser.add_argument("--future-horizon-s", type=float, default=30.0)

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
    if args.command == "sweep-pyrecest":
        return _sweep_pyrecest(args)
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
        models=_parse_models(args.models),
        **_pyrecest_scalar_kwargs(args),
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
    rows = []
    config = BenchmarkConfig(
        candidate_top_k=args.candidate_top_k,
        models=_parse_models(args.models),
        **_pyrecest_scalar_kwargs(args),
    )
    for model in _build_models(config, session=session).values():
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
        **_pyrecest_scalar_kwargs(args),
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


def _sweep_pyrecest(args: argparse.Namespace) -> int:
    config = PyRecEstSweepConfig(
        max_events_per_session=args.max_events,
        candidate_top_k=args.candidate_top_k,
        random_seed=args.random_seed,
        event_epoch=args.event_epoch,
        baseline_models=_parse_optional_models(args.baseline_models),
        pyrecest_models=_parse_models(args.pyrecest_models),
        particles=_parse_int_values(args.particles),
        alphas=_parse_float_values(args.alpha),
        betas=_parse_float_values(args.beta),
        process_noise_sigmas_cm_s=_parse_float_values(args.process_noise_sigma_cm_s),
        position_jump_sigmas_cm=_parse_float_values(args.position_jump_sigma_cm),
        jump_probabilities=_parse_float_values(args.jump_probability),
        goal_reset_probabilities=_parse_float_values(args.goal_reset_probability),
        initial_velocity_sigmas_cm_s=_parse_float_values(
            args.initial_velocity_sigma_cm_s
        ),
        imm_mode_stickinesses=_parse_float_values(args.imm_mode_stickiness),
        imm_stationary_velocity_decays=_parse_float_values(
            args.imm_stationary_velocity_decay
        ),
        imm_diffusion_velocity_decays=_parse_float_values(
            args.imm_diffusion_velocity_decay
        ),
        imm_momentum_velocity_decays=_parse_float_values(
            args.imm_momentum_velocity_decay
        ),
        imm_jump_fractions=_parse_float_values(args.imm_jump_fraction),
        imm_jump_velocity_decays=_parse_float_values(args.imm_jump_velocity_decay),
        include_ground_truth=not args.skip_ground_truth,
        ground_truth=_ground_truth_config_from_args(args),
    )
    result = run_pyrecest_parameter_sweep(args.root, config)
    write_pyrecest_sweep_outputs(result, args.output)
    print(pareto_sweep_summary(result.summary).to_string(index=False))
    return 0


def _add_pyrecest_scalar_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pyrecest-particles", type=int, default=512)
    parser.add_argument("--pyrecest-alpha", type=float, default=0.80)
    parser.add_argument("--pyrecest-beta", type=float, default=1.00)
    parser.add_argument("--pyrecest-process-noise-sigma-cm-s", type=float, default=60.0)
    parser.add_argument("--pyrecest-position-jump-sigma-cm", type=float, default=25.0)
    parser.add_argument("--pyrecest-jump-probability", type=float, default=0.03)
    parser.add_argument("--pyrecest-goal-reset-probability", type=float, default=0.02)
    parser.add_argument(
        "--pyrecest-initial-velocity-sigma-cm-s",
        type=float,
        default=120.0,
    )
    parser.add_argument("--pyrecest-imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument(
        "--pyrecest-imm-stationary-velocity-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--pyrecest-imm-diffusion-velocity-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--pyrecest-imm-momentum-velocity-decay",
        type=float,
        default=0.95,
    )
    parser.add_argument("--pyrecest-imm-jump-fraction", type=float, default=0.9)
    parser.add_argument("--pyrecest-imm-jump-velocity-decay", type=float, default=0.25)


def _pyrecest_scalar_kwargs(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "pyrecest_particles": args.pyrecest_particles,
        "pyrecest_alpha": args.pyrecest_alpha,
        "pyrecest_beta": args.pyrecest_beta,
        "pyrecest_process_noise_sigma_cm_s": args.pyrecest_process_noise_sigma_cm_s,
        "pyrecest_position_jump_sigma_cm": args.pyrecest_position_jump_sigma_cm,
        "pyrecest_jump_probability": args.pyrecest_jump_probability,
        "pyrecest_goal_reset_probability": args.pyrecest_goal_reset_probability,
        "pyrecest_initial_velocity_sigma_cm_s": args.pyrecest_initial_velocity_sigma_cm_s,
        "pyrecest_imm_mode_stickiness": args.pyrecest_imm_mode_stickiness,
        "pyrecest_imm_stationary_velocity_decay": (
            args.pyrecest_imm_stationary_velocity_decay
        ),
        "pyrecest_imm_diffusion_velocity_decay": (
            args.pyrecest_imm_diffusion_velocity_decay
        ),
        "pyrecest_imm_momentum_velocity_decay": (
            args.pyrecest_imm_momentum_velocity_decay
        ),
        "pyrecest_imm_jump_fraction": args.pyrecest_imm_jump_fraction,
        "pyrecest_imm_jump_velocity_decay": args.pyrecest_imm_jump_velocity_decay,
    }


def _parse_int_values(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _split_csv_values(value))


def _parse_float_values(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _split_csv_values(value))


def _split_csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated value list must contain at least one value")
    return values


def _parse_models(value: str) -> tuple[str, ...]:
    models = tuple(model.strip() for model in value.split(",") if model.strip())
    if not models:
        raise ValueError("--models must contain at least one model name")
    return models


def _parse_optional_models(value: str) -> tuple[str, ...]:
    if value.strip().lower() in {"", "none"}:
        return ()
    return _parse_models(value)


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
