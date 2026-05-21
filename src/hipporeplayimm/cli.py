"""Command-line interface for HippoReplayIMM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmarks import (
    BenchmarkConfig,
    _build_models,
    _clusterless_mark_config,
    _is_clusterless_model,
    bootstrap_delta_ci,
    run_open_field_benchmark,
)
from .clusterless import (
    ClusterlessMarkConfig,
    build_clusterless_mark_emissions,
    clusterless_mark_likelihood_label,
    fit_clusterless_mark_encoding,
)
from .data import load_open_field_sessions
from .encoding import EmissionConfig, EncodingConfig, build_emissions, fit_place_field_encoding
from .ground_truth import (
    GroundTruthConfig,
    GroundTruthSensitivityConfig,
    compare_scores_to_ground_truth,
    compare_scores_to_ground_truth_sensitivity,
    generate_behavioral_ground_truth,
)
from .observation_sweep import (
    ObservationSweepConfig,
    run_observation_parameter_sweep,
    write_observation_sweep_outputs,
)
from .position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_DECODE_BIN_S,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
    PositionDecodingConfig,
    run_position_decoding_validation,
)
from .simulation_recovery import (
    DEFAULT_SCORING_MODELS,
    DEFAULT_TRUE_MODELS,
    SimulationRecoveryConfig,
    parse_model_list,
    run_session_simulation_recovery,
)
from .result_improvements import write_benchmark_settings
from .state_space import StateSpaceDecoderConfig, StateSpaceReplayModel
from .sweeps import (
    PyRecEstSweepConfig,
    pareto_aggregate_sweep_summary,
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
    benchmark_parser.add_argument(
        "--n-cell-splits",
        type=int,
        default=1,
        help="Number of independent held-out cell splits to score.",
    )
    benchmark_parser.add_argument(
        "--randomize-event-subset",
        action="store_true",
        help="Sample --max-events without replacement instead of taking the first events.",
    )
    benchmark_parser.add_argument("--event-subset-seed", type=int)
    benchmark_parser.add_argument("--candidate-top-k", type=int, default=64)
    benchmark_parser.add_argument("--test-cell-fraction", type=float, default=0.25)
    benchmark_parser.add_argument("--random-seed", type=int, default=1)
    benchmark_parser.add_argument(
        "--random-seeds",
        help="Comma-separated train/test cell split seeds; overrides --random-seed when set.",
    )
    benchmark_parser.add_argument("--time-bin-ms", type=float, default=20.0)
    benchmark_parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    _add_emission_calibration_arguments(benchmark_parser)
    benchmark_parser.add_argument(
        "--clusterless-mark-group-by",
        choices=("auto", "none", "tetrode", "cell"),
        default="auto",
        help="Clusterless mark-likelihood grouping. 'auto' uses tetrode groups when Tetrode_Cell_IDs are available.",
    )
    benchmark_parser.add_argument(
        "--models",
        default="random,stationary,diffusion,momentum,imm",
        help="Comma-separated model names to benchmark.",
    )
    _add_encoding_arguments(benchmark_parser)
    _add_state_space_arguments(benchmark_parser)
    _add_clusterless_arguments(benchmark_parser)
    _add_pyrecest_scalar_arguments(benchmark_parser)

    decode_parser = subparsers.add_parser("decode-event")
    decode_parser.add_argument("root")
    decode_parser.add_argument("--session", required=True)
    decode_parser.add_argument("--event-id", type=int, required=True)
    decode_parser.add_argument("--candidate-top-k", type=int, default=64)
    decode_parser.add_argument("--time-bin-ms", type=float, default=20.0)
    decode_parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    _add_emission_calibration_arguments(decode_parser)
    decode_parser.add_argument("--output")
    decode_parser.add_argument(
        "--models",
        default="random,stationary,diffusion,momentum,imm",
        help="Comma-separated model names to score.",
    )
    _add_encoding_arguments(decode_parser)
    _add_state_space_arguments(decode_parser)
    _add_clusterless_arguments(decode_parser)
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
    compare_parser.add_argument("--test-cell-fraction", type=float, default=0.25)
    compare_parser.add_argument("--random-seed", type=int, default=1)
    compare_parser.add_argument("--time-bin-ms", type=float, default=20.0)
    compare_parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    _add_emission_calibration_arguments(compare_parser)
    _add_encoding_arguments(compare_parser)
    _add_state_space_arguments(compare_parser)
    _add_clusterless_arguments(compare_parser)
    _add_pyrecest_scalar_arguments(compare_parser)
    compare_parser.add_argument("--visit-radius-cm", type=float, default=10.0)
    compare_parser.add_argument("--min-dwell-s", type=float, default=0.2)
    compare_parser.add_argument("--future-horizon-s", type=float, default=30.0)

    sensitivity_parser = subparsers.add_parser("ground-truth-sensitivity")
    sensitivity_parser.add_argument("root")
    sensitivity_parser.add_argument("--scores", required=True)
    sensitivity_parser.add_argument("--output", required=True)
    sensitivity_parser.add_argument("--candidate-top-k", type=int, default=64)
    sensitivity_parser.add_argument("--test-cell-fraction", type=float, default=0.25)
    sensitivity_parser.add_argument("--random-seed", type=int, default=1)
    sensitivity_parser.add_argument("--time-bin-ms", type=float, default=20.0)
    sensitivity_parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    _add_emission_calibration_arguments(sensitivity_parser)
    _add_encoding_arguments(sensitivity_parser)
    _add_state_space_arguments(sensitivity_parser)
    _add_clusterless_arguments(sensitivity_parser)
    _add_pyrecest_scalar_arguments(sensitivity_parser)
    sensitivity_parser.add_argument("--visit-radii-cm", default="7.5,10.0,12.5")
    sensitivity_parser.add_argument("--min-dwells-s", default="0.1,0.2,0.4")
    sensitivity_parser.add_argument("--future-horizons-s", default="15.0,30.0,60.0")
    sensitivity_parser.add_argument("--well-arrival-window-s", type=float, default=1.0)
    sensitivity_parser.add_argument(
        "--event-epoch",
        choices=("run", "all"),
        default="run",
    )

    validate_parser = subparsers.add_parser("validate-position")
    validate_parser.add_argument("root")
    validate_parser.add_argument("--output", required=True)
    validate_parser.add_argument("--session")
    validate_parser.add_argument("--decode-bin-s", type=float, default=VALIDATED_POSITION_DECODE_BIN_S)
    validate_parser.add_argument("--n-folds", type=int, default=5)
    validate_parser.add_argument("--max-windows", type=int)
    validate_parser.add_argument("--random-seed", type=int, default=1)
    validate_parser.add_argument("--min-spikes-per-window", type=int, default=0)
    validate_parser.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    validate_parser.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    validate_parser.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    validate_parser.add_argument("--min-occupancy-s", type=float, default=EncodingConfig().min_occupancy_s)
    validate_parser.add_argument("--rate-floor-hz", type=float, default=EncodingConfig().rate_floor_hz)

    observation_parser = subparsers.add_parser("sweep-observation")
    observation_parser.add_argument("root")
    observation_parser.add_argument("--output", required=True)
    observation_parser.add_argument(
        "--sessions",
        default="Rat1/Open1",
        help="Comma-separated session IDs, or 'all' for all loaded sessions.",
    )
    observation_parser.add_argument("--decode-bin-s", type=float, default=VALIDATED_POSITION_DECODE_BIN_S)
    observation_parser.add_argument("--n-folds", type=int, default=5)
    observation_parser.add_argument("--max-windows", type=int)
    observation_parser.add_argument("--min-spikes-per-window", type=int, default=0)
    observation_parser.add_argument("--random-seed", type=int, default=1)
    observation_parser.add_argument("--bin-size-cm", default=f"{VALIDATED_POSITION_BIN_SIZE_CM:g}")
    observation_parser.add_argument("--smoothing-sigma-bins", default=f"{VALIDATED_POSITION_SMOOTHING_SIGMA_BINS:g}")
    observation_parser.add_argument("--min-speed-cm-s", default=f"{VALIDATED_POSITION_MIN_SPEED_CM_S:g}")
    observation_parser.add_argument("--min-occupancy-s", default=f"{EncodingConfig().min_occupancy_s:g}")
    observation_parser.add_argument("--rate-floor-hz", default=f"{EncodingConfig().rate_floor_hz:g}")
    observation_parser.add_argument(
        "--time-bin-ms",
        default="3.0",
        help="Comma-separated replay bin widths, in milliseconds, for synthetic recovery.",
    )
    observation_parser.add_argument(
        "--spike-rate-scale",
        default="0.5,1.0,2.0",
        help="Comma-separated replay spike-rate multipliers for synthetic recovery.",
    )
    observation_parser.add_argument(
        "--emission-likelihood-temperature",
        default="1.0,1.5,2.0",
        help="Comma-separated emission likelihood temperatures for synthetic recovery scoring.",
    )
    observation_parser.add_argument(
        "--emission-negative-binomial-overdispersion",
        default="0.0,0.03,0.1",
        help="Comma-separated negative-binomial overdispersion values for synthetic recovery scoring.",
    )
    observation_parser.add_argument("--skip-simulation-recovery", action="store_true")
    observation_parser.add_argument("--simulation-events", default="run")
    observation_parser.add_argument("--simulation-max-template-events", type=int, default=25)
    observation_parser.add_argument("--simulation-events-per-model", type=int, default=10)
    observation_parser.add_argument("--simulation-true-models", default=" ".join(DEFAULT_TRUE_MODELS))
    observation_parser.add_argument("--simulation-models", default=" ".join(DEFAULT_SCORING_MODELS))
    observation_parser.add_argument("--simulation-continue-on-error", action="store_true")

    recovery_parser = subparsers.add_parser("simulate-recovery")
    recovery_parser.add_argument("root")
    recovery_parser.add_argument("--session", required=True)
    recovery_parser.add_argument("--output", required=True)
    recovery_parser.add_argument("--events", default="run")
    recovery_parser.add_argument("--max-template-events", type=int, default=25)
    recovery_parser.add_argument("--events-per-model", type=int, default=25)
    recovery_parser.add_argument("--true-models", default=" ".join(DEFAULT_TRUE_MODELS))
    recovery_parser.add_argument("--models", default=" ".join(DEFAULT_SCORING_MODELS))
    recovery_parser.add_argument("--random-seed", type=int, default=1)
    recovery_parser.add_argument("--time-bin-ms", type=float, default=3.0)
    recovery_parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    _add_emission_calibration_arguments(recovery_parser)
    recovery_parser.add_argument("--state-space-sigma-cm-sqrt-s", type=float, default=85.0)
    recovery_parser.add_argument("--state-space-stationary-sigma-cm", type=float, default=2.0)
    recovery_parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=None)
    recovery_parser.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    recovery_parser.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)
    recovery_parser.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=None)
    recovery_parser.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=None)
    recovery_parser.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.95)
    recovery_parser.add_argument("--state-space-momentum-candidate-top-k", type=int, default=128)
    recovery_parser.add_argument("--state-space-momentum-candidate-mass-threshold", type=float, default=None)
    recovery_parser.add_argument("--state-space-momentum-candidate-min-k", type=int, default=1)
    recovery_parser.add_argument("--state-space-momentum-candidate-max-k", type=int, default=0)
    recovery_parser.add_argument("--state-space-momentum-predicted-candidate-top-k", type=int, default=StateSpaceDecoderConfig().momentum_predicted_candidate_top_k)
    recovery_parser.add_argument("--candidate-top-k", type=int, default=64)
    recovery_parser.add_argument("--stationary-sigma-cm", type=float, default=2.0)
    recovery_parser.add_argument("--diffusion-sigma-cm", type=float, default=12.0)
    recovery_parser.add_argument("--momentum-sigma-cm", type=float, default=12.0)
    recovery_parser.add_argument("--velocity-decay", type=float, default=0.95)
    recovery_parser.add_argument("--mode-stickiness", type=float, default=0.94)
    recovery_parser.add_argument("--continue-on-error", action="store_true")
    _add_encoding_arguments(recovery_parser)

    sweep_parser = subparsers.add_parser("sweep-pyrecest")
    sweep_parser.add_argument("root")
    sweep_parser.add_argument("--output", required=True)
    sweep_parser.add_argument("--max-events", type=int, default=1)
    sweep_parser.add_argument("--candidate-top-k", type=int, default=64)
    sweep_parser.add_argument("--random-seed", type=int, default=1)
    sweep_parser.add_argument(
        "--random-seeds",
        help="Comma-separated random seeds to run; overrides --random-seed.",
    )
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
    sweep_parser.add_argument("--position-proposal-probability", default="0.0")
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
    if args.command == "ground-truth-sensitivity":
        return _ground_truth_sensitivity(args)
    if args.command == "validate-position":
        return _validate_position(args)
    if args.command == "sweep-observation":
        return _sweep_observation(args)
    if args.command == "simulate-recovery":
        return _simulate_recovery(args)
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
            "spike_mark_features": 0 if session.spike_marks is None else session.spike_marks.n_features,
            "spike_mark_source": "" if session.spike_marks is None else f"{session.spike_marks.source_file}:{session.spike_marks.source_variable}",
            "clusterless_mark_likelihood": clusterless_mark_likelihood_label(session),
            "ripples": session.ripple_count,
            "run_ripples": session.ripple_indices_in_run().shape[0],
        }
        for session in sessions
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


def _validate_position(args: argparse.Namespace) -> int:
    config = PositionDecodingConfig(
        encoding=EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
            min_occupancy_s=args.min_occupancy_s,
            rate_floor_hz=args.rate_floor_hz,
        ),
        decode_bin_s=args.decode_bin_s,
        n_folds=args.n_folds,
        max_windows_per_session=args.max_windows,
        random_seed=args.random_seed,
        min_spikes_per_window=args.min_spikes_per_window,
        session=args.session,
    )
    result = run_position_decoding_validation(args.root, config)
    result.write(args.output)
    print(result.summary.to_string(index=False))
    return 0


def _sweep_observation(args: argparse.Namespace) -> int:
    simulation_max_template_events = (
        None
        if args.simulation_max_template_events is not None
        and args.simulation_max_template_events <= 0
        else args.simulation_max_template_events
    )
    sessions = (
        None
        if args.sessions.strip().lower() == "all"
        else _parse_string_values(args.sessions)
    )
    config = ObservationSweepConfig(
        sessions=sessions,
        bin_sizes_cm=_parse_float_values(args.bin_size_cm),
        smoothing_sigmas_bins=_parse_float_values(args.smoothing_sigma_bins),
        min_speed_cm_s=_parse_float_values(args.min_speed_cm_s),
        min_occupancy_s=_parse_float_values(args.min_occupancy_s),
        rate_floor_hz=_parse_float_values(args.rate_floor_hz),
        time_bin_ms=_parse_float_values(args.time_bin_ms),
        spike_rate_scales=_parse_float_values(args.spike_rate_scale),
        likelihood_temperatures=_parse_float_values(
            args.emission_likelihood_temperature
        ),
        negative_binomial_overdispersions=_parse_float_values(
            args.emission_negative_binomial_overdispersion
        ),
        decode_bin_s=args.decode_bin_s,
        n_folds=args.n_folds,
        max_windows_per_session=args.max_windows,
        min_spikes_per_window=args.min_spikes_per_window,
        random_seed=args.random_seed,
        run_simulation_recovery=not args.skip_simulation_recovery,
        simulation_events=args.simulation_events,
        simulation_max_template_events=simulation_max_template_events,
        simulation_events_per_model=args.simulation_events_per_model,
        simulation_true_models=parse_model_list(args.simulation_true_models),
        simulation_scoring_models=parse_model_list(args.simulation_models),
        simulation_continue_on_error=args.simulation_continue_on_error,
    )
    result = run_observation_parameter_sweep(args.root, config)
    write_observation_sweep_outputs(result, args.output)
    print(result.summary.to_string(index=False))
    return 0


def _simulate_recovery(args: argparse.Namespace) -> int:
    shared_sigma = args.state_space_sigma_cm_sqrt_s
    state_space = StateSpaceDecoderConfig(
        stationary_sigma_cm=args.state_space_stationary_sigma_cm,
        diffusion_sigma_cm_sqrt_s=args.state_space_diffusion_sigma_cm_sqrt_s or shared_sigma,
        max_step_sigma=args.state_space_max_step_sigma,
        imm_mode_stickiness=args.state_space_imm_mode_stickiness,
        momentum_sigma_cm_sqrt_s=args.state_space_momentum_sigma_cm_sqrt_s or shared_sigma,
        momentum_initial_sigma_cm_sqrt_s=args.state_space_momentum_initial_sigma_cm_sqrt_s or shared_sigma,
        momentum_velocity_decay=args.state_space_momentum_velocity_decay,
        momentum_candidate_top_k=args.state_space_momentum_candidate_top_k,
        momentum_candidate_mass_threshold=args.state_space_momentum_candidate_mass_threshold,
        momentum_candidate_min_k=args.state_space_momentum_candidate_min_k,
        momentum_candidate_max_k=args.state_space_momentum_candidate_max_k,
        momentum_predicted_candidate_top_k=args.state_space_momentum_predicted_candidate_top_k,
    )
    config = SimulationRecoveryConfig(
        true_models=parse_model_list(args.true_models),
        scoring_models=parse_model_list(args.models),
        events=args.events,
        max_template_events=args.max_template_events,
        events_per_model=args.events_per_model,
        random_seed=args.random_seed,
        spike_rate_scale=args.spike_rate_scale,
        time_bin_s=args.time_bin_ms / 1000.0,
        likelihood_temperature=args.emission_likelihood_temperature,
        negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
        encoding=_encoding_config_from_args(args),
        state_space=state_space,
        candidate_top_k=args.candidate_top_k,
        stationary_sigma_cm=args.stationary_sigma_cm,
        diffusion_sigma_cm=args.diffusion_sigma_cm,
        momentum_sigma_cm=args.momentum_sigma_cm,
        velocity_decay=args.velocity_decay,
        mode_stickiness=args.mode_stickiness,
        continue_on_error=args.continue_on_error,
    )
    result = run_session_simulation_recovery(args.root, args.session, config)
    result.write(args.output)
    print(result.summary.to_string(index=False))
    print("\nConfusion matrix:")
    print(result.confusion_matrix.to_string(index=False))
    print(f"\nRows: {len(result.event_scores)}")
    print(f"Failures: {int((result.event_scores['status'] != 'success').sum())}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    if args.preset != "open-field-loso":
        raise ValueError("Only --preset open-field-loso is currently implemented")
    config = BenchmarkConfig(
        encoding=_encoding_config_from_args(args),
        emissions=_emission_config_from_args(args),
        max_events_per_session=args.max_events,
        n_cell_splits=args.n_cell_splits,
        randomize_event_subset=args.randomize_event_subset,
        event_subset_seed=args.event_subset_seed,
        test_cell_fraction=args.test_cell_fraction,
        random_seed=args.random_seed,
        random_seeds=_parse_int_values(args.random_seeds) if args.random_seeds else None,
        candidate_top_k=args.candidate_top_k,
        models=_parse_models(args.models),
        clusterless_mark_group_by=args.clusterless_mark_group_by,
        **_state_space_scalar_kwargs(args),
        **_clusterless_scalar_kwargs(args),
        **_pyrecest_scalar_kwargs(args),
    )
    result = run_open_field_benchmark(args.root, config)
    print(result.summary().to_string(index=False))
    if not result.rows.empty and "imm" in set(result.rows["model"]):
        value_column = "delta_vs_best_static"
        label = "IMM mean delta vs exact best static bootstrap 95% CI"
        imm_mask = result.rows["model"].eq("imm")
        if (
            result.rows.loc[imm_mask, value_column].dropna().empty
            and "lower_bound_delta_vs_best_static" in result.rows
        ):
            value_column = "lower_bound_delta_vs_best_static"
            label = "IMM lower-bound mean delta vs exact best static bootstrap 95% CI"
        ci_low, ci_high = bootstrap_delta_ci(result.rows, model="imm", value_column=value_column)
        if np.isfinite(ci_low) and np.isfinite(ci_high):
            print(f"{label}: [{ci_low:.3f}, {ci_high:.3f}]")
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        result.rows.to_csv(output / "event_scores.csv", index=False)
        result.summary().to_csv(output / "summary.csv", index=False)
        result.session_summary().to_csv(output / "summary_by_session.csv", index=False)
        result.rat_summary().to_csv(output / "summary_by_rat.csv", index=False)
        result.split_summary().to_csv(output / "summary_by_split.csv", index=False)
        write_benchmark_settings(output / "benchmark_settings.yml", config, vars(args), result.rows)
    return 0


def _decode_event(args: argparse.Namespace) -> int:
    sessions = {session.session_id: session for session in load_open_field_sessions(args.root)}
    if args.session not in sessions:
        available = ", ".join(sorted(sessions))
        raise KeyError(f"Unknown session {args.session!r}; available: {available}")
    session = sessions[args.session]

    encoding_config = _encoding_config_from_args(args)
    emission_config = _emission_config_from_args(args)
    requested_models = _parse_models(args.models)
    config = BenchmarkConfig(
        encoding=encoding_config,
        emissions=emission_config,
        candidate_top_k=args.candidate_top_k,
        models=requested_models,
        **_state_space_scalar_kwargs(args),
        **_clusterless_scalar_kwargs(args),
        **_pyrecest_scalar_kwargs(args),
    )
    model_objects = _build_models(config, session=session)
    has_clusterless_models = any(_is_clusterless_model(model) for model in model_objects.values())
    has_sorted_spike_models = any(
        not _is_clusterless_model(model) for model in model_objects.values()
    )

    sorted_encoding = None
    sorted_emissions = None
    if has_sorted_spike_models:
        sorted_encoding = fit_place_field_encoding(session, encoding_config)
        sorted_emissions = build_emissions(
            session,
            sorted_encoding,
            args.event_id,
            emission_config,
        )

    clusterless_encoding = None
    clusterless_emissions = None
    if has_clusterless_models:
        clusterless_encoding = fit_clusterless_mark_encoding(
            session,
            _clusterless_mark_config(config),
        )
        clusterless_emissions = build_clusterless_mark_emissions(
            session,
            clusterless_encoding,
            args.event_id,
            emission_config,
        )

    rows = []
    posterior_artifacts: list[tuple[str, object, object, object]] = []
    for model in model_objects.values():
        if _is_clusterless_model(model):
            assert clusterless_emissions is not None
            assert clusterless_encoding is not None
            model_emissions = clusterless_emissions
            model_encoding = clusterless_encoding
        else:
            assert sorted_emissions is not None
            assert sorted_encoding is not None
            model_emissions = sorted_emissions
            model_encoding = sorted_encoding

        if isinstance(model, StateSpaceReplayModel):
            score = model.score(
                model_emissions,
                model_encoding.bin_centers,
                occupancy_s=model_encoding.occupancy_s,
            )
        else:
            score = model.score(model_emissions, model_encoding.bin_centers)
        rows.append(
            {
                "model": score.model_name,
                "log_likelihood": score.log_likelihood,
                "n_spikes": score.n_spikes,
                **score.diagnostics,
            }
        )
        if score.trajectory_log_posterior is not None:
            posterior_artifacts.append(
                (
                    score.model_name,
                    score.trajectory_log_posterior,
                    model_emissions,
                    model_encoding,
                )
            )
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output / "event_scores.csv", index=False)
        safe_session = args.session.replace("/", "_").replace("\\", "_")
        for (
            model_name,
            trajectory_log_posterior,
            model_emissions,
            model_encoding,
        ) in posterior_artifacts:
            safe_model = model_name.replace("/", "_").replace("\\", "_")
            cell_ids = getattr(model_encoding, "cell_ids", model_emissions.cell_ids)
            np.savez_compressed(
                output / f"{safe_session}_event{int(args.event_id):04d}_{safe_model}_posterior.npz",
                log_posteriors=np.asarray(trajectory_log_posterior, dtype=float),
                trajectory_log_posteriors=np.asarray(trajectory_log_posterior, dtype=float),
                times=model_emissions.times,
                bin_centers=model_encoding.bin_centers,
                x_edges=model_encoding.x_edges,
                y_edges=model_encoding.y_edges,
                grid_shape=np.asarray(model_encoding.grid_shape, dtype=int),
                cell_ids=cell_ids,
                spike_counts=model_emissions.spike_counts,
            )
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
        encoding_config=_encoding_config_from_args(args),
        emission_config=_emission_config_from_args(args),
        test_cell_fraction=args.test_cell_fraction,
        candidate_top_k=args.candidate_top_k,
        random_seed=args.random_seed,
        **_state_space_scalar_kwargs(args),
        **_clusterless_scalar_kwargs(args),
        **_pyrecest_scalar_kwargs(args),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(_comparison_summary(frame).to_string(index=False))
    return 0


def _ground_truth_sensitivity(args: argparse.Namespace) -> int:
    config = GroundTruthSensitivityConfig(
        visit_radii_cm=_parse_float_values(args.visit_radii_cm),
        min_dwells_s=_parse_float_values(args.min_dwells_s),
        future_horizons_s=_parse_float_values(args.future_horizons_s),
        well_arrival_window_s=args.well_arrival_window_s,
        event_epoch=args.event_epoch,
    )
    result = compare_scores_to_ground_truth_sensitivity(
        args.root,
        args.scores,
        sensitivity_config=config,
        encoding_config=_encoding_config_from_args(args),
        emission_config=_emission_config_from_args(args),
        test_cell_fraction=args.test_cell_fraction,
        candidate_top_k=args.candidate_top_k,
        random_seed=args.random_seed,
        **_state_space_scalar_kwargs(args),
        **_clusterless_scalar_kwargs(args),
        **_pyrecest_scalar_kwargs(args),
    )
    result.write(args.output)
    if result.robustness_summary.empty:
        print("No valid behavioral labels under any sensitivity setting.")
    else:
        print(result.robustness_summary.to_string(index=False))
    return 0


def _ground_truth_config_from_args(args: argparse.Namespace) -> GroundTruthConfig:
    return GroundTruthConfig(
        visit_radius_cm=args.visit_radius_cm,
        min_dwell_s=args.min_dwell_s,
        future_horizon_s=args.future_horizon_s,
    )


def _encoding_config_from_args(args: argparse.Namespace) -> EncodingConfig:
    defaults = EncodingConfig()
    return EncodingConfig(
        bin_size_cm=args.bin_size_cm,
        smoothing_sigma_bins=args.smoothing_sigma_bins,
        min_speed_cm_s=args.min_speed_cm_s,
        min_occupancy_s=getattr(args, "min_occupancy_s", defaults.min_occupancy_s),
        rate_floor_hz=getattr(args, "rate_floor_hz", defaults.rate_floor_hz),
    )


def _emission_config_from_args(args: argparse.Namespace) -> EmissionConfig:
    return EmissionConfig(
        time_bin_s=args.time_bin_ms / 1000.0,
        spike_rate_scale=args.spike_rate_scale,
        likelihood_temperature=args.emission_likelihood_temperature,
        negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
    )


def _add_state_space_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = StateSpaceDecoderConfig()
    parser.add_argument(
        "--state-space-stationary-sigma-cm",
        type=float,
        default=defaults.stationary_sigma_cm,
    )
    parser.add_argument(
        "--state-space-diffusion-sigma-cm-sqrt-s",
        type=float,
        default=defaults.diffusion_sigma_cm_sqrt_s,
    )
    parser.add_argument(
        "--state-space-max-step-sigma",
        type=float,
        default=defaults.max_step_sigma,
    )
    parser.add_argument(
        "--state-space-imm-mode-stickiness",
        type=float,
        default=defaults.imm_mode_stickiness,
    )
    parser.add_argument(
        "--state-space-momentum-sigma-cm-sqrt-s",
        type=float,
        default=defaults.momentum_sigma_cm_sqrt_s,
    )
    parser.add_argument(
        "--state-space-momentum-initial-sigma-cm-sqrt-s",
        type=float,
        default=defaults.momentum_initial_sigma_cm_sqrt_s,
    )
    parser.add_argument(
        "--state-space-momentum-velocity-decay",
        type=float,
        default=defaults.momentum_velocity_decay,
    )
    parser.add_argument(
        "--state-space-momentum-candidate-top-k",
        type=int,
        default=defaults.momentum_candidate_top_k,
    )
    parser.add_argument(
        "--state-space-momentum-candidate-mass-threshold",
        type=float,
        default=defaults.momentum_candidate_mass_threshold,
    )
    parser.add_argument(
        "--state-space-momentum-candidate-min-k",
        type=int,
        default=defaults.momentum_candidate_min_k,
    )
    parser.add_argument(
        "--state-space-momentum-candidate-max-k",
        type=int,
        default=defaults.momentum_candidate_max_k,
    )
    parser.add_argument(
        "--state-space-momentum-predicted-candidate-top-k",
        type=int,
        default=defaults.momentum_predicted_candidate_top_k,
    )
    parser.add_argument(
        "--state-space-valid-occupancy-threshold-s",
        type=float,
        default=defaults.valid_occupancy_threshold_s,
        help=(
            "If positive, state-space priors and transition normalizers are restricted "
            "to spatial bins whose training occupancy is at least this many seconds."
        ),
    )


def _add_clusterless_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clusterless-mark-smoothing-sigma-bins", type=float, default=1.0)
    parser.add_argument("--clusterless-mark-prior-count", type=float, default=1.0)
    parser.add_argument("--clusterless-mark-variance-floor", type=float, default=1.0)
    parser.add_argument("--clusterless-rate-floor-hz", type=float, default=1e-4)
    parser.add_argument("--clusterless-mark-likelihood", choices=("local-kde", "diagonal-gaussian"), default="local-kde")
    parser.add_argument("--clusterless-mark-kde-bandwidth", type=float, default=None)
    parser.add_argument("--clusterless-mark-kde-spatial-sigma-bins", type=float, default=None)
    parser.add_argument("--clusterless-mark-kde-max-neighbors", type=int, default=256)


def _clusterless_mark_config_from_args(args: argparse.Namespace) -> ClusterlessMarkConfig:
    return ClusterlessMarkConfig(
        encoding=_encoding_config_from_args(args),
        mark_smoothing_sigma_bins=args.clusterless_mark_smoothing_sigma_bins,
        mark_prior_count=args.clusterless_mark_prior_count,
        mark_variance_floor=args.clusterless_mark_variance_floor,
        rate_floor_hz=args.clusterless_rate_floor_hz,
        mark_likelihood=args.clusterless_mark_likelihood,
        mark_kde_bandwidth=args.clusterless_mark_kde_bandwidth,
        mark_kde_spatial_sigma_bins=args.clusterless_mark_kde_spatial_sigma_bins,
        mark_kde_max_neighbors=args.clusterless_mark_kde_max_neighbors,
    )


def _sweep_pyrecest(args: argparse.Namespace) -> int:
    config = PyRecEstSweepConfig(
        max_events_per_session=args.max_events,
        candidate_top_k=args.candidate_top_k,
        random_seed=args.random_seed,
        random_seeds=(
            _parse_int_values(args.random_seeds)
            if args.random_seeds
            else (args.random_seed,)
        ),
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
        position_proposal_probabilities=_parse_float_values(
            args.position_proposal_probability
        ),
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
    if result.aggregate_summary.empty:
        print(pareto_sweep_summary(result.summary).to_string(index=False))
    else:
        print(
            pareto_aggregate_sweep_summary(result.aggregate_summary).to_string(
                index=False
            )
        )
    return 0


def _add_encoding_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = EncodingConfig()
    parser.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    parser.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    parser.add_argument("--min-occupancy-s", type=float, default=defaults.min_occupancy_s)
    parser.add_argument("--rate-floor-hz", type=float, default=defaults.rate_floor_hz)


def _add_emission_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--emission-likelihood-temperature",
        type=float,
        default=1.0,
        help="Divide emission log likelihoods by this positive temperature; values >1 flatten the emission model.",
    )
    parser.add_argument(
        "--emission-negative-binomial-overdispersion",
        type=float,
        default=0.0,
        help="Use a negative-binomial count model with variance mean + alpha * mean**2; 0 keeps the Poisson model.",
    )


def _add_pyrecest_scalar_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pyrecest-particles", type=int, default=512)
    parser.add_argument("--pyrecest-alpha", type=float, default=0.80)
    parser.add_argument("--pyrecest-beta", type=float, default=1.00)
    parser.add_argument("--pyrecest-process-noise-sigma-cm-s", type=float, default=60.0)
    parser.add_argument("--pyrecest-position-jump-sigma-cm", type=float, default=25.0)
    parser.add_argument("--pyrecest-jump-probability", type=float, default=0.03)
    parser.add_argument("--pyrecest-goal-reset-probability", type=float, default=0.02)
    parser.add_argument(
        "--pyrecest-position-proposal-probability",
        type=float,
        default=0.0,
    )
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
        "pyrecest_position_proposal_probability": (
            args.pyrecest_position_proposal_probability
        ),
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


def _state_space_scalar_kwargs(args: argparse.Namespace) -> dict[str, float | int | None]:
    return {
        "state_space_valid_occupancy_threshold_s": args.state_space_valid_occupancy_threshold_s,
        "state_space_stationary_sigma_cm": args.state_space_stationary_sigma_cm,
        "state_space_diffusion_sigma_cm_sqrt_s": args.state_space_diffusion_sigma_cm_sqrt_s,
        "state_space_max_step_sigma": args.state_space_max_step_sigma,
        "state_space_imm_mode_stickiness": args.state_space_imm_mode_stickiness,
        "state_space_momentum_sigma_cm_sqrt_s": args.state_space_momentum_sigma_cm_sqrt_s,
        "state_space_momentum_initial_sigma_cm_sqrt_s": (
            args.state_space_momentum_initial_sigma_cm_sqrt_s
        ),
        "state_space_momentum_velocity_decay": args.state_space_momentum_velocity_decay,
        "state_space_momentum_candidate_top_k": args.state_space_momentum_candidate_top_k,
        "state_space_momentum_candidate_mass_threshold": args.state_space_momentum_candidate_mass_threshold,
        "state_space_momentum_candidate_min_k": args.state_space_momentum_candidate_min_k,
        "state_space_momentum_candidate_max_k": args.state_space_momentum_candidate_max_k,
        "state_space_momentum_predicted_candidate_top_k": args.state_space_momentum_predicted_candidate_top_k,
    }


def _clusterless_scalar_kwargs(args: argparse.Namespace) -> dict[str, float | int | str | None]:
    return {
        "clusterless_mark_smoothing_sigma_bins": args.clusterless_mark_smoothing_sigma_bins,
        "clusterless_mark_prior_count": args.clusterless_mark_prior_count,
        "clusterless_mark_variance_floor": args.clusterless_mark_variance_floor,
        "clusterless_rate_floor_hz": args.clusterless_rate_floor_hz,
        "clusterless_mark_likelihood": args.clusterless_mark_likelihood,
        "clusterless_mark_kde_bandwidth": args.clusterless_mark_kde_bandwidth,
        "clusterless_mark_kde_spatial_sigma_bins": args.clusterless_mark_kde_spatial_sigma_bins,
        "clusterless_mark_kde_max_neighbors": args.clusterless_mark_kde_max_neighbors,
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


def _parse_string_values(value: str) -> tuple[str, ...]:
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
    agg_spec = {
        "rows": ("event_index", "count"),
        "goal_accuracy": ("goal_correct", "mean"),
        "median_endpoint_error_cm": ("endpoint_error_cm", "median"),
        "mean_true_well_posterior": ("true_well_posterior", "mean"),
    }
    if "active_goal_correct" in valid.columns:
        agg_spec["active_goal_accuracy"] = ("active_goal_correct", "mean")
    return (
        valid.groupby("model", as_index=False)
        .agg(**agg_spec)
        .sort_values("model")
    )


if __name__ == "__main__":
    raise SystemExit(main())
