#!/usr/bin/env python3
"""Improved opt-in replay model-evidence benchmark.

This script keeps the existing ``benchmark_model_evidence.py`` intact and adds
an opt-in entry point that exposes the result-improvement controls that are most
useful for full-session evidence runs:

* exact goal-conditioned state-space models;
* reverse-time and bidirectional hypothesis wrappers;
* physical-time IMM mode stickiness via a switching time constant;
* adaptive second-order candidate augmentation;
* clusterless local-KDE controls;
* replay-specific emission gain and overdispersion calibration;
* spatial-bin shuffle null controls;
* model-averaged endpoint summaries.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_model_evidence import (  # reuse the stable writer/reporting helpers
    _check_session,
    _counts,
    _events,
    _session_path,
    _summary,
    _write,
    _postprocess_evidence_scores,
)

from hipporeplayimm.accuracy_upgrades import (
    ValidStateConfig,
    ValidStateDiffusionReplayModel,
    ValidStateGridReplayModel,
    candidate_event_windows,
    event_reliability_flags,
    valid_state_mask_from_encoding,
)
from hipporeplayimm.clusterless import (
    ClusterlessMarkConfig,
    ClusterlessStateSpaceReplayModel,
    build_clusterless_mark_emissions,
    fit_clusterless_mark_encoding,
)
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, fit_place_field_encoding
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel
from hipporeplayimm.ground_truth import infer_well_locations
from hipporeplayimm.models import CandidateKinematicModel, RandomModel, StationaryModel
from hipporeplayimm.position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
)
from hipporeplayimm.result_improvement_extensions import (
    BidirectionalReplayModel,
    ReplayEmissionCalibration,
    ReverseTimeReplayModel,
    add_model_averaged_endpoint_columns,
    build_sorted_emissions_with_replay_calibration,
    score_replay_model_compat,
    score_spatial_shuffle_nulls,
)
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig

_TRAJECTORY_MODELS = {
    "diffusion",
    "momentum",
    "imm",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-jump",
    "sorted-spike-state-space-momentum",
    "sorted-spike-state-space-momentum-reverse",
    "sorted-spike-state-space-momentum-bidirectional",
    "sorted-spike-state-space-momentum-exact-sparse",
    "sorted-spike-state-space-trajectory-imm-exact-sparse",
    "sorted-spike-state-space-imm",
    "sorted-spike-state-space-goal",
    "sorted-spike-state-space-goal-reverse",
    "sorted-spike-state-space-goal-bidirectional",
    "state-space-goal",
    "clusterless-state-space-diffusion",
    "clusterless-state-space-fragmented",
    "clusterless-state-space-jump",
    "clusterless-state-space-momentum",
    "clusterless-state-space-momentum-exact-sparse",
    "clusterless-state-space-trajectory-imm-exact-sparse",
    "clusterless-state-space-imm",
    "valid-state-diffusion",
    "valid-state-grid",
}
_NONTRAJECTORY_MODELS = {
    "random",
    "stationary",
    "stationary-gaussian",
    "sorted-spike-state-space-stationary",
    "clusterless-state-space-stationary",
}
_ALIASES = {
    "stationary_gaussian": "stationary-gaussian",
    "state-space-goal-forward": "state-space-goal",
    "sorted-spike-state-space-goal-forward": "sorted-spike-state-space-goal",
}
DEFAULT_IMPROVED_MODELS = (
    "random stationary sorted-spike-state-space-diffusion "
    "sorted-spike-state-space-momentum-exact-sparse "
    "sorted-spike-state-space-momentum sorted-spike-state-space-imm "
    "sorted-spike-state-space-goal sorted-spike-state-space-goal-bidirectional"
)
DEFAULT_IMPROVED_STATE_SPACE_IMM_SWITCH_TAU_S = 0.060
DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_CANDIDATE_TOP_K = 256
DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_PREDICTED_CANDIDATE_TOP_K = 16


def _family(model: str) -> str:
    if model in _TRAJECTORY_MODELS:
        return "trajectory"
    if model in _NONTRAJECTORY_MODELS:
        return "nontrajectory"
    return "other"


def _effective_state_space_imm_stickiness(args: argparse.Namespace) -> float:
    tau = float(args.state_space_imm_switch_tau_s)
    if tau <= 0.0:
        return float(args.state_space_imm_mode_stickiness)
    return float(np.exp(-float(args.time_bin_s) / tau))


def _parse_float_sequence(value: str) -> tuple[float, ...]:
    items = []
    for raw in str(value).replace(",", " ").split():
        text = raw.strip()
        if text:
            items.append(float(text))
    if not items:
        raise ValueError("expected at least one floating-point value")
    return tuple(items)


def _parse_window_variant_specs(value: str) -> tuple[tuple[str, float, float], ...]:
    """Parse NAME:PRE_PAD_S:POST_PAD_S replay-window variants."""

    variants: list[tuple[str, float, float]] = []
    for raw in str(value).replace(",", " ").split():
        text = raw.strip()
        if not text:
            continue
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(
                "window variant specs must use NAME:PRE_PAD_S:POST_PAD_S entries"
            )
        name, pre_pad, post_pad = parts
        if not name:
            raise ValueError("window variant names must be non-empty")
        variants.append((name, float(pre_pad), float(post_pad)))
    if not variants:
        raise ValueError("expected at least one window variant spec")
    return tuple(variants)


def _event_windows(args: argparse.Namespace, event) -> pd.DataFrame:
    if args.window_variant_specs:
        rows = []
        for name, pre_pad, post_pad in _parse_window_variant_specs(args.window_variant_specs):
            start = float(event.start) - float(pre_pad)
            end = float(event.end) + float(post_pad)
            duration = end - start
            if duration >= float(args.window_min_duration_s):
                rows.append(
                    {
                        "event_window_variant": name,
                        "pre_pad_s": float(pre_pad),
                        "post_pad_s": float(post_pad),
                        "window_start_s": start,
                        "window_end_s": end,
                        "window_duration_s": duration,
                    }
                )
        return pd.DataFrame(rows)

    windows = candidate_event_windows(
        event,
        pre_pads_s=_parse_float_sequence(args.window_pre_pads_s),
        post_pads_s=_parse_float_sequence(args.window_post_pads_s),
        min_duration_s=args.window_min_duration_s,
    )
    if not windows.empty:
        windows = windows.copy()
        windows["event_window_variant"] = [
            f"pre{pre_pad:+.3f}_post{post_pad:+.3f}"
            for pre_pad, post_pad in zip(
                windows["pre_pad_s"], windows["post_pad_s"], strict=True
            )
        ]
    return windows


def _optional_threshold(value: float) -> float | None:
    return None if not np.isfinite(float(value)) else float(value)


def _state_space_config(args: argparse.Namespace, mode: str) -> StateSpaceDecoderConfig:
    return StateSpaceDecoderConfig(
        mode=mode,
        stationary_sigma_cm=args.state_space_stationary_sigma_cm,
        diffusion_sigma_cm_sqrt_s=args.state_space_diffusion_sigma_cm_sqrt_s,
        max_step_sigma=args.state_space_max_step_sigma,
        imm_mode_stickiness=_effective_state_space_imm_stickiness(args),
        momentum_sigma_cm_sqrt_s=args.state_space_momentum_sigma_cm_sqrt_s,
        momentum_initial_sigma_cm_sqrt_s=args.state_space_momentum_initial_sigma_cm_sqrt_s,
        momentum_velocity_decay=args.state_space_momentum_velocity_decay,
        momentum_velocity_decay_tau_s=args.state_space_momentum_velocity_decay_tau_s,
        momentum_candidate_top_k=args.state_space_momentum_candidate_top_k,
        momentum_candidate_mass_threshold=args.state_space_momentum_candidate_mass_threshold,
        momentum_candidate_min_k=args.state_space_momentum_candidate_min_k,
        momentum_candidate_max_k=args.state_space_momentum_candidate_max_k,
        momentum_predicted_candidate_top_k=args.state_space_momentum_predicted_candidate_top_k,
        momentum_candidate_source=args.state_space_momentum_candidate_source,
        valid_occupancy_threshold_s=args.state_space_valid_occupancy_threshold_s,
    )


def _goal_candidates(session) -> np.ndarray | None:
    wells = infer_well_locations(session)
    if wells.empty:
        return None
    return wells[["well_x", "well_y"]].to_numpy(dtype=float)


def _models(args: argparse.Namespace, session, encoding=None) -> dict[str, object]:
    names = []
    for raw in args.models.replace(",", " ").split():
        name = _ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if name:
            names.append(name)
    if args.include_clusterless_defaults:
        names.extend(
            [
                "clusterless-state-space-diffusion",
                "clusterless-state-space-momentum",
                "clusterless-state-space-imm",
            ]
        )
    if not names:
        raise ValueError("no models selected")

    wants_goal_state_space = any(
        name in {
            "sorted-spike-state-space-goal",
            "sorted-spike-state-space-goal-reverse",
            "sorted-spike-state-space-goal-bidirectional",
            "state-space-goal",
        }
        for name in names
    )
    goal_candidates = _goal_candidates(session) if wants_goal_state_space else None
    valid_mask = None
    valid_grid_shape = (1, 1)
    if encoding is not None:
        valid_mask = valid_state_mask_from_encoding(
            encoding,
            ValidStateConfig(
                min_occupancy_s=args.valid_state_min_occupancy_s,
                keep_top_occupancy_fraction=args.valid_state_top_occupancy_fraction,
            ),
        )
        valid_grid_shape = encoding.grid_shape
    valid_mask_safe = np.ones(1, dtype=bool) if valid_mask is None else valid_mask

    def state_space_model(mode: str, *, name: str | None = None) -> SortedSpikeStateSpaceReplayModel:
        return SortedSpikeStateSpaceReplayModel(
            mode=mode,
            config=_state_space_config(args, mode),
            name=name,
        )

    def clusterless_model(mode: str) -> ClusterlessStateSpaceReplayModel:
        return ClusterlessStateSpaceReplayModel(
            mode=mode,
            config=_state_space_config(args, mode),
            mark_likelihood=args.clusterless_mark_likelihood,
        )

    def goal_model(name: str) -> GoalStateSpaceReplayModel:
        return GoalStateSpaceReplayModel(
            candidate_goals=goal_candidates,
            transition_sigma_cm_sqrt_s=args.goal_state_space_transition_sigma_cm_sqrt_s,
            drift_speed_cm_s=args.goal_state_space_drift_speed_cm_s,
            max_step_sigma=args.goal_state_space_max_step_sigma,
            name=name,
        )

    forward_goal = goal_model("sorted-spike-state-space-goal")
    reverse_goal = ReverseTimeReplayModel(
        goal_model("sorted-spike-state-space-goal"),
        name="sorted-spike-state-space-goal-reverse",
    )
    forward_momentum = state_space_model("momentum")
    reverse_momentum = ReverseTimeReplayModel(
        state_space_model("momentum"),
        name="sorted-spike-state-space-momentum-reverse",
    )

    available = {
        "random": RandomModel(),
        "stationary": StationaryModel(),
        "stationary-gaussian": CandidateKinematicModel(
            mode="stationary",
            top_k=args.candidate_top_k,
            stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm,
            momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay,
            mode_stickiness=args.mode_stickiness,
            name="stationary-gaussian",
        ),
        "diffusion": CandidateKinematicModel(
            mode="diffusion",
            top_k=args.candidate_top_k,
            stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm,
            momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay,
            mode_stickiness=args.mode_stickiness,
            name="diffusion",
        ),
        "momentum": CandidateKinematicModel(
            mode="momentum",
            top_k=args.candidate_top_k,
            stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm,
            momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay,
            mode_stickiness=args.mode_stickiness,
            name="momentum",
        ),
        "imm": CandidateKinematicModel(
            mode="imm",
            top_k=args.candidate_top_k,
            stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm,
            momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay,
            mode_stickiness=args.mode_stickiness,
            name="imm",
        ),
        "sorted-spike-state-space-stationary": state_space_model("stationary"),
        "sorted-spike-state-space-diffusion": state_space_model("diffusion"),
        "sorted-spike-state-space-fragmented": state_space_model("fragmented"),
        "sorted-spike-state-space-first-order-imm": state_space_model("first-order-imm"),
        "sorted-spike-state-space-jump": state_space_model("jump"),
        "sorted-spike-state-space-momentum": forward_momentum,
        "sorted-spike-state-space-momentum-reverse": reverse_momentum,
        "sorted-spike-state-space-momentum-exact-sparse": state_space_model("momentum-exact-sparse"),
        "sorted-spike-state-space-trajectory-imm-exact-sparse": state_space_model("trajectory-imm-exact-sparse"),
        "sorted-spike-state-space-momentum-bidirectional": BidirectionalReplayModel(
            forward_momentum,
            reverse_momentum,
            name="sorted-spike-state-space-momentum-bidirectional",
        ),
        "sorted-spike-state-space-imm": state_space_model("imm"),
        "sorted-spike-state-space-goal": forward_goal,
        "state-space-goal": goal_model("state-space-goal"),
        "sorted-spike-state-space-goal-reverse": reverse_goal,
        "sorted-spike-state-space-goal-bidirectional": BidirectionalReplayModel(
            forward_goal,
            reverse_goal,
            name="sorted-spike-state-space-goal-bidirectional",
        ),
        "clusterless-state-space-stationary": clusterless_model("stationary"),
        "clusterless-state-space-diffusion": clusterless_model("diffusion"),
        "clusterless-state-space-fragmented": clusterless_model("fragmented"),
        "clusterless-state-space-jump": clusterless_model("jump"),
        "clusterless-state-space-momentum": clusterless_model("momentum"),
        "clusterless-state-space-momentum-exact-sparse": clusterless_model("momentum-exact-sparse"),
        "clusterless-state-space-trajectory-imm-exact-sparse": clusterless_model("trajectory-imm-exact-sparse"),
        "clusterless-state-space-imm": clusterless_model("imm"),
        "valid-state-diffusion": ValidStateDiffusionReplayModel(
            valid_mask_safe,
            sigma_cm=args.valid_state_sigma_cm,
            max_step_sigma=args.valid_state_max_step_sigma,
            name="valid-state-diffusion",
        ),
        "valid-state-grid": ValidStateGridReplayModel(
            valid_mask_safe,
            grid_shape=valid_grid_shape,
            diagonal_neighbors=bool(args.valid_state_grid_diagonal_neighbors),
            stay_probability=float(args.valid_state_grid_stay_probability),
            name="valid-state-grid",
        ),
    }
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"unknown models: {missing}; available: {sorted(available)}")
    return {name: available[name] for name in dict.fromkeys(names)}


def _clusterless_mark_config(args: argparse.Namespace) -> ClusterlessMarkConfig:
    return ClusterlessMarkConfig(
        encoding=EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
            min_occupancy_s=args.min_occupancy_s,
            rate_floor_hz=args.rate_floor_hz,
        ),
        mark_smoothing_sigma_bins=args.clusterless_mark_smoothing_sigma_bins,
        mark_prior_count=args.clusterless_mark_prior_count,
        mark_variance_floor=args.clusterless_mark_variance_floor,
        rate_floor_hz=args.clusterless_rate_floor_hz,
        mark_likelihood=args.clusterless_mark_likelihood,
        mark_kde_bandwidth=args.clusterless_mark_kde_bandwidth,
        mark_kde_spatial_sigma_bins=args.clusterless_mark_kde_spatial_sigma_bins,
        mark_kde_max_neighbors=args.clusterless_mark_kde_max_neighbors,
        mark_group_by=args.clusterless_mark_group_by,
    )


def _score(args: argparse.Namespace) -> pd.DataFrame:
    session_dir = _session_path(args.dataset_root, args.session)
    _check_session(session_dir)
    session = load_replay_session(session_dir)
    event_ids = _events(args.events, session)
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]

    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
            min_occupancy_s=args.min_occupancy_s,
            rate_floor_hz=args.rate_floor_hz,
        ),
    )
    models = _models(args, session, encoding=encoding)
    has_clusterless = any(isinstance(model, ClusterlessStateSpaceReplayModel) for model in models.values())
    clusterless_encoding = None
    if has_clusterless:
        clusterless_encoding = fit_clusterless_mark_encoding(session, _clusterless_mark_config(args))

    emissions_cfg = EmissionConfig(
        time_bin_s=args.time_bin_s,
        spike_rate_scale=args.spike_rate_scale,
        likelihood_temperature=args.emission_likelihood_temperature,
        negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
    )
    sorted_calibration = ReplayEmissionCalibration(
        gain_mode=args.replay_gain_mode,
        gain_prior_count=args.replay_gain_prior_count,
        max_gain=args.replay_gain_max_gain,
        emission_model=args.sorted_spike_emission_model,
        negative_binomial_dispersion=args.negative_binomial_dispersion,
    )
    rows: list[dict[str, object]] = []

    for event_id in event_ids:
        windows = _event_windows(args, session.ripple(int(event_id)))
        for window_index, window in windows.reset_index(drop=True).iterrows():
            event_window = SimpleNamespace(
                start=float(window["window_start_s"]),
                end=float(window["window_end_s"]),
            )
            window_settings = {
                "window_index": int(window_index),
                "window_pre_pad_s": float(window["pre_pad_s"]),
                "window_post_pad_s": float(window["post_pad_s"]),
                "event_window_variant": str(window["event_window_variant"]),
                "window_start_s": float(window["window_start_s"]),
                "window_end_s": float(window["window_end_s"]),
                "window_duration_s": float(window["window_duration_s"]),
            }
            sorted_emissions = build_sorted_emissions_with_replay_calibration(
                session,
                encoding,
                event_window,
                emissions_cfg,
                calibration=sorted_calibration,
            )
            clusterless_emissions = (
                build_clusterless_mark_emissions(session, clusterless_encoding, event_window, emissions_cfg)
                if clusterless_encoding is not None
                else None
            )
            if sorted_emissions.n_time == 0:
                continue
            for name, model in models.items():
                start = time.perf_counter()
                use_clusterless = isinstance(model, ClusterlessStateSpaceReplayModel)
                emissions = clusterless_emissions if use_clusterless else sorted_emissions
                bin_centers = (
                    clusterless_encoding.bin_centers
                    if use_clusterless and clusterless_encoding is not None
                    else encoding.bin_centers
                )
                assert emissions is not None
                occupancy_s = (
                    clusterless_encoding.occupancy_s
                    if use_clusterless and clusterless_encoding is not None
                    else encoding.occupancy_s
                )
                try:
                    result = score_replay_model_compat(model, emissions, bin_centers, occupancy_s=occupancy_s)
                    model_name = str(result.model_name)
                    row: dict[str, object] = {
                        "status": "success",
                        "session": session.session_id,
                        "event_index": int(event_id),
                        **window_settings,
                        "model": model_name,
                        "requested_model": name,
                        "model_family": _family(model_name),
                        "log_evidence": float(result.log_likelihood),
                        "n_time": int(result.n_time),
                        "n_spikes": int(result.n_spikes),
                        "runtime_s": float(time.perf_counter() - start),
                        "error": "",
                        **_run_settings(args),
                    }
                    metadata = getattr(emissions, "metadata", {}) or {}
                    row.update({f"emission_{key}": value for key, value in metadata.items()})
                    if use_clusterless and clusterless_encoding is not None:
                        row.update(
                            {
                                "clusterless_mark_features": int(clusterless_encoding.n_features),
                                "clusterless_spike_mark_source": clusterless_encoding.spike_mark_source,
                            }
                        )
                    row.update({f"diagnostic_{key}": value for key, value in result.diagnostics.items()})
                    row.update(
                        event_reliability_flags(
                            row,
                            min_spikes=args.reliability_min_spikes,
                            min_time_bins=args.reliability_min_time_bins,
                            max_entropy=_optional_threshold(args.reliability_max_terminal_entropy),
                            min_candidate_log_mass=_optional_threshold(args.reliability_min_candidate_log_mass),
                        )
                    )
                    if args.null_shuffles > 0:
                        null_seed = int(
                            args.null_random_seed
                            + 1009 * int(event_id)
                            + 313 * int(window_index)
                            + 7919 * list(models).index(name)
                        )
                        row.update(
                            {
                                f"spatial_shuffle_{key}": value
                                for key, value in score_spatial_shuffle_nulls(
                                    model,
                                    emissions,
                                    bin_centers,
                                    observed_log_evidence=float(result.log_likelihood),
                                    n_shuffles=args.null_shuffles,
                                    random_seed=null_seed,
                                    occupancy_s=occupancy_s,
                                ).items()
                            }
                        )
                    rows.append(row)
                    print(f"Scored {session.session_id} event {event_id} window {window_index} with {name}", flush=True)
                except Exception as exc:
                    rows.append(
                        {
                            "status": "failure",
                            "session": session.session_id,
                            "event_index": int(event_id),
                            **window_settings,
                            "model": name,
                            "requested_model": name,
                            "model_family": _family(name),
                            "log_evidence": np.nan,
                            "n_time": int(emissions.n_time),
                            "n_spikes": int(emissions.n_spikes),
                            "runtime_s": float(time.perf_counter() - start),
                            "error": f"{type(exc).__name__}: {exc}",
                            **_run_settings(args),
                        }
                    )
                    if not args.continue_on_error:
                        raise
    return add_model_averaged_endpoint_columns(_postprocess_evidence_scores(pd.DataFrame(rows)))


def _run_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "bin_size_cm": float(args.bin_size_cm),
        "smoothing_sigma_bins": float(args.smoothing_sigma_bins),
        "min_speed_cm_s": float(args.min_speed_cm_s),
        "min_occupancy_s": float(args.min_occupancy_s),
        "rate_floor_hz": float(args.rate_floor_hz),
        "time_bin_s": float(args.time_bin_s),
        "spike_rate_scale": float(args.spike_rate_scale),
        "emission_likelihood_temperature": float(args.emission_likelihood_temperature),
        "emission_negative_binomial_overdispersion": float(args.emission_negative_binomial_overdispersion),
        "sorted_spike_emission_model": args.sorted_spike_emission_model,
        "replay_gain_mode": args.replay_gain_mode,
        "replay_gain_prior_count": float(args.replay_gain_prior_count),
        "replay_gain_max_gain": float(args.replay_gain_max_gain),
        "negative_binomial_dispersion": float(args.negative_binomial_dispersion),
        "candidate_top_k": int(args.candidate_top_k),
        "state_space_valid_occupancy_threshold_s": float(args.state_space_valid_occupancy_threshold_s),
        "state_space_stationary_sigma_cm": float(args.state_space_stationary_sigma_cm),
        "state_space_diffusion_sigma_cm_sqrt_s": float(args.state_space_diffusion_sigma_cm_sqrt_s),
        "state_space_max_step_sigma": float(args.state_space_max_step_sigma),
        "state_space_imm_mode_stickiness_input": float(args.state_space_imm_mode_stickiness),
        "state_space_imm_switch_tau_s": float(args.state_space_imm_switch_tau_s),
        "state_space_imm_mode_stickiness_effective": _effective_state_space_imm_stickiness(args),
        "state_space_momentum_sigma_cm_sqrt_s": float(args.state_space_momentum_sigma_cm_sqrt_s),
        "state_space_momentum_initial_sigma_cm_sqrt_s": float(args.state_space_momentum_initial_sigma_cm_sqrt_s),
        "state_space_momentum_velocity_decay": float(args.state_space_momentum_velocity_decay),
        "state_space_momentum_velocity_decay_tau_s": float(args.state_space_momentum_velocity_decay_tau_s),
        "state_space_momentum_candidate_top_k": int(args.state_space_momentum_candidate_top_k),
        "state_space_momentum_candidate_mass_threshold": "" if args.state_space_momentum_candidate_mass_threshold is None else float(args.state_space_momentum_candidate_mass_threshold),
        "state_space_momentum_candidate_min_k": int(args.state_space_momentum_candidate_min_k),
        "state_space_momentum_candidate_max_k": int(args.state_space_momentum_candidate_max_k),
        "state_space_momentum_predicted_candidate_top_k": int(args.state_space_momentum_predicted_candidate_top_k),
        "state_space_momentum_candidate_source": str(args.state_space_momentum_candidate_source),
        "clusterless_mark_likelihood": args.clusterless_mark_likelihood,
        "clusterless_mark_group_by": args.clusterless_mark_group_by,
        "clusterless_mark_smoothing_sigma_bins": float(args.clusterless_mark_smoothing_sigma_bins),
        "clusterless_mark_prior_count": float(args.clusterless_mark_prior_count),
        "clusterless_mark_variance_floor": float(args.clusterless_mark_variance_floor),
        "clusterless_rate_floor_hz": float(args.clusterless_rate_floor_hz),
        "clusterless_mark_kde_bandwidth": "" if args.clusterless_mark_kde_bandwidth is None else float(args.clusterless_mark_kde_bandwidth),
        "clusterless_mark_kde_spatial_sigma_bins": "" if args.clusterless_mark_kde_spatial_sigma_bins is None else float(args.clusterless_mark_kde_spatial_sigma_bins),
        "clusterless_mark_kde_max_neighbors": int(args.clusterless_mark_kde_max_neighbors),
        "goal_state_space_transition_sigma_cm_sqrt_s": float(args.goal_state_space_transition_sigma_cm_sqrt_s),
        "goal_state_space_drift_speed_cm_s": float(args.goal_state_space_drift_speed_cm_s),
        "goal_state_space_max_step_sigma": float(args.goal_state_space_max_step_sigma),
        "include_clusterless_defaults": bool(args.include_clusterless_defaults),
        "valid_state_min_occupancy_s": float(args.valid_state_min_occupancy_s),
        "valid_state_top_occupancy_fraction": "" if args.valid_state_top_occupancy_fraction is None else float(args.valid_state_top_occupancy_fraction),
        "valid_state_sigma_cm": float(args.valid_state_sigma_cm),
        "valid_state_max_step_sigma": float(args.valid_state_max_step_sigma),
        "valid_state_grid_diagonal_neighbors": bool(args.valid_state_grid_diagonal_neighbors),
        "valid_state_grid_stay_probability": float(args.valid_state_grid_stay_probability),
        "window_variant_specs": str(args.window_variant_specs),
        "window_pre_pads_s": str(args.window_pre_pads_s),
        "window_post_pads_s": str(args.window_post_pads_s),
        "window_min_duration_s": float(args.window_min_duration_s),
        "reliability_min_spikes": int(args.reliability_min_spikes),
        "reliability_min_time_bins": int(args.reliability_min_time_bins),
        "reliability_max_terminal_entropy": float(args.reliability_max_terminal_entropy),
        "reliability_min_candidate_log_mass": float(args.reliability_min_candidate_log_mass),
        "null_shuffles": int(args.null_shuffles),
        "null_random_seed": int(args.null_random_seed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the improved replay model-evidence benchmark.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="run:0-25")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument(
        "--models",
        default=DEFAULT_IMPROVED_MODELS,
    )
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument("--stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--diffusion-sigma-cm", type=float, default=12.0)
    parser.add_argument("--momentum-sigma-cm", type=float, default=12.0)
    parser.add_argument("--velocity-decay", type=float, default=0.95)
    parser.add_argument("--mode-stickiness", type=float, default=0.94)
    parser.add_argument("--state-space-stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument("--state-space-imm-switch-tau-s", type=float, default=DEFAULT_IMPROVED_STATE_SPACE_IMM_SWITCH_TAU_S)
    parser.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.95)
    parser.add_argument("--state-space-momentum-velocity-decay-tau-s", type=float, default=0.0)
    parser.add_argument("--state-space-momentum-candidate-top-k", type=int, default=DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_CANDIDATE_TOP_K)
    parser.add_argument("--state-space-momentum-candidate-mass-threshold", type=float)
    parser.add_argument("--state-space-momentum-candidate-min-k", type=int, default=1)
    parser.add_argument("--state-space-momentum-candidate-max-k", type=int, default=0)
    parser.add_argument("--state-space-momentum-predicted-candidate-top-k", type=int, default=DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_PREDICTED_CANDIDATE_TOP_K)
    parser.add_argument("--state-space-momentum-candidate-source", choices=("emission", "posterior"), default="emission")
    parser.add_argument("--state-space-valid-occupancy-threshold-s", type=float, default=0.0)
    parser.add_argument("--goal-state-space-transition-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--goal-state-space-drift-speed-cm-s", type=float, default=400.0)
    parser.add_argument("--goal-state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--clusterless-mark-likelihood", default="local-kde")
    parser.add_argument("--clusterless-mark-group-by", choices=("auto", "none", "tetrode", "cell"), default="auto")
    parser.add_argument("--clusterless-mark-smoothing-sigma-bins", type=float, default=1.0)
    parser.add_argument("--clusterless-mark-prior-count", type=float, default=1.0)
    parser.add_argument("--clusterless-mark-variance-floor", type=float, default=1.0)
    parser.add_argument("--clusterless-rate-floor-hz", type=float, default=1e-4)
    parser.add_argument("--clusterless-mark-kde-bandwidth", type=float)
    parser.add_argument("--clusterless-mark-kde-spatial-sigma-bins", type=float)
    parser.add_argument("--clusterless-mark-kde-max-neighbors", type=int, default=256)
    parser.add_argument("--time-bin-s", type=float, default=0.003)
    parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    parser.add_argument("--emission-likelihood-temperature", type=float, default=1.0)
    parser.add_argument("--emission-negative-binomial-overdispersion", type=float, default=0.0)
    parser.add_argument("--sorted-spike-emission-model", choices=("poisson", "negative-binomial", "gamma-poisson"), default="poisson")
    parser.add_argument("--replay-gain-mode", choices=("none", "event", "cell", "event-cell"), default="none")
    parser.add_argument("--replay-gain-prior-count", type=float, default=10.0)
    parser.add_argument("--replay-gain-max-gain", type=float, default=20.0)
    parser.add_argument("--negative-binomial-dispersion", type=float, default=50.0)
    parser.add_argument("--include-clusterless-defaults", action="store_true")
    parser.add_argument("--valid-state-min-occupancy-s", type=float, default=0.02)
    parser.add_argument("--valid-state-top-occupancy-fraction", type=float, default=None)
    parser.add_argument("--valid-state-sigma-cm", type=float, default=5.0)
    parser.add_argument("--valid-state-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--valid-state-grid-diagonal-neighbors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--valid-state-grid-stay-probability", type=float, default=0.0)
    parser.add_argument(
        "--window-variant-specs",
        default="",
        help=(
            "Optional named replay-window variants as "
            "NAME:PRE_PAD_S:POST_PAD_S entries separated by spaces or commas."
        ),
    )
    parser.add_argument("--window-pre-pads-s", default="0.0")
    parser.add_argument("--window-post-pads-s", default="0.0")
    parser.add_argument("--window-min-duration-s", type=float, default=0.005)
    parser.add_argument("--reliability-min-spikes", type=int, default=5)
    parser.add_argument("--reliability-min-time-bins", type=int, default=2)
    parser.add_argument("--reliability-max-terminal-entropy", type=float, default=float("nan"))
    parser.add_argument("--reliability-min-candidate-log-mass", type=float, default=-0.01)
    parser.add_argument("--null-shuffles", type=int, default=0)
    parser.add_argument("--null-random-seed", type=int, default=1)
    parser.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    parser.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    parser.add_argument("--min-occupancy-s", type=float, default=EncodingConfig().min_occupancy_s)
    parser.add_argument("--rate-floor-hz", type=float, default=EncodingConfig().rate_floor_hz)
    parser.add_argument("--output", default="results/model-evidence-improved")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    df = _score(args)
    if df.empty:
        raise RuntimeError("No scores were generated.")
    print(_summary(df).to_string(index=False))
    print("\nBest-model counts:")
    print(_counts(df).to_string(index=False))
    print(f"\nRows: {len(df)}")
    print(f"Failures: {int((df['status'] != 'success').sum())}")
    _write(df, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
