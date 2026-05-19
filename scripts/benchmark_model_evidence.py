#!/usr/bin/env python3
"""Session-scoped full-event replay model-evidence benchmark.

This is an approximate model-evidence diagnostic using the repository's current
model ``score`` methods. It is meant as a positive-control step toward the
Krause/Drugowitsch-style Bayesian model-comparison result, not as an exact
reproduction of their Zenodo analysis code.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.clusterless import (
    ClusterlessMarkConfig,
    ClusterlessStateSpaceReplayModel,
    build_clusterless_mark_emissions,
    fit_clusterless_mark_encoding,
)
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import (
    EmissionConfig,
    EncodingConfig,
    build_emissions,
    fit_place_field_encoding,
)
from hipporeplayimm.evidence_reporting import (
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns as _ensure_evidence_support_columns,
)
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel
from hipporeplayimm.goal_state_space_integration import (
    DEFAULT_GOAL_COMPONENT_SWITCH_PROBABILITY,
    DEFAULT_GOAL_DIFFUSION_MIXTURE_WEIGHT,
    DEFAULT_GOAL_DRIFT_SPEED_CM_S,
    DEFAULT_GOAL_FORWARD_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT,
    DEFAULT_GOAL_INITIAL_POSITION_PRIOR_DIRECTION_MODE,
    DEFAULT_GOAL_INITIAL_PRIOR_SIGMA_CM,
    DEFAULT_GOAL_INITIAL_PRIOR_WEIGHT,
    DEFAULT_GOAL_LATERAL_SIGMA_SCALE,
    DEFAULT_GOAL_MAX_STEP_SIGMA,
    DEFAULT_GOAL_REVERSE_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT,
    DEFAULT_GOAL_REVERSE_TERMINAL_POSITION_PRIOR_WEIGHT,
    DEFAULT_GOAL_RESET_INITIAL_POSITION_PRIOR_WEIGHT,
    DEFAULT_GOAL_RESET_PROBABILITY,
    DEFAULT_GOAL_SWITCHING_COMPONENT_SWITCH_PROBABILITY,
    DEFAULT_GOAL_TERMINAL_PRIOR_SIGMA_CM,
    DEFAULT_GOAL_TERMINAL_PRIOR_WEIGHT,
    DEFAULT_GOAL_TOWARD_DIRECTION_PRIOR_WEIGHT,
    DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S,
)
from hipporeplayimm.ground_truth import active_goal_at_time, infer_well_locations
from hipporeplayimm.models import CandidateKinematicModel, RandomModel, StationaryModel
from hipporeplayimm.position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
)
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig

_REQUIRED = ("Position_Data.mat", "Ripple_Events.mat", "Spike_Data.mat", "Epochs.mat")
_TRAJ = {
    "diffusion",
    "momentum",
    "imm",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-jump",
    "sorted-spike-state-space-goal",
    "sorted-spike-state-space-goal-bidirectional",
    "sorted-spike-state-space-goal-forward-biased",
    "sorted-spike-state-space-goal-forward-biased-switching",
    "sorted-spike-state-space-goal-reverse-biased",
    "sorted-spike-state-space-momentum",
    "sorted-spike-state-space-imm",
    "state-space-goal",
    "state-space-goal-bidirectional",
    "state-space-goal-forward-biased",
    "state-space-goal-forward-biased-switching",
    "state-space-goal-reverse-biased",
    "clusterless-state-space-diffusion",
    "clusterless-state-space-fragmented",
    "clusterless-state-space-jump",
    "clusterless-state-space-momentum",
    "clusterless-state-space-imm",
}
_NONTRAJ = {
    "random",
    "stationary",
    "stationary-gaussian",
    "sorted-spike-state-space-stationary",
    "clusterless-state-space-stationary",
}
_ALIASES = {"stationary_gaussian": "stationary-gaussian"}


def _session_path(root: str | Path, session: str) -> Path:
    parts = session.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("session must have the form 'RatN/OpenM', e.g. 'Rat1/Open1'")
    return Path(root) / parts[0] / parts[1]


def _check_session(path: Path) -> None:
    missing = [name for name in _REQUIRED if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Requested session {path} is missing: {', '.join(missing)}")


def _ints(spec: str) -> list[int]:
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = [int(x) for x in item.split("-", 1)]
            if hi < lo:
                raise ValueError(f"descending range: {item}")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("no events selected")
    return sorted(dict.fromkeys(values))


def _events(spec: str, session) -> list[int]:
    s = spec.strip().lower()
    if s == "all":
        return list(range(session.ripple_count))
    if s == "run":
        return [int(x) for x in session.ripple_indices_in_run()]
    if s.startswith("run:"):
        run = [int(x) for x in session.ripple_indices_in_run()]
        out = []
        for ordinal in _ints(s.split(":", 1)[1]):
            if ordinal < 0 or ordinal >= len(run):
                raise IndexError(f"run ordinal {ordinal} outside 0..{len(run) - 1}")
            out.append(run[ordinal])
        return sorted(dict.fromkeys(out))
    out = _ints(spec)
    bad = [e for e in out if e < 0 or e >= session.ripple_count]
    if bad:
        raise IndexError(f"event IDs outside 0..{session.ripple_count - 1}: {bad}")
    return out


def _models(args, session=None) -> dict[str, object]:
    names = []
    for raw in args.models.replace(",", " ").split():
        name = _ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if name:
            names.append(name)
    if not names:
        raise ValueError("no models selected")

    def state_space_model(mode: str) -> SortedSpikeStateSpaceReplayModel:
        return SortedSpikeStateSpaceReplayModel(
            mode=mode,
            config=StateSpaceDecoderConfig(
                mode=mode,
                stationary_sigma_cm=args.state_space_stationary_sigma_cm,
                diffusion_sigma_cm_sqrt_s=args.state_space_diffusion_sigma_cm_sqrt_s,
                max_step_sigma=args.state_space_max_step_sigma,
                imm_mode_stickiness=args.state_space_imm_mode_stickiness,
                momentum_sigma_cm_sqrt_s=args.state_space_momentum_sigma_cm_sqrt_s,
                momentum_initial_sigma_cm_sqrt_s=args.state_space_momentum_initial_sigma_cm_sqrt_s,
                momentum_velocity_decay=args.state_space_momentum_velocity_decay,
                momentum_candidate_top_k=args.state_space_momentum_candidate_top_k,
            ),
        )

    def clusterless_state_space_model(mode: str) -> ClusterlessStateSpaceReplayModel:
        return ClusterlessStateSpaceReplayModel(
            mode=mode,
            config=StateSpaceDecoderConfig(
                mode=mode,
                stationary_sigma_cm=args.state_space_stationary_sigma_cm,
                diffusion_sigma_cm_sqrt_s=args.state_space_diffusion_sigma_cm_sqrt_s,
                max_step_sigma=args.state_space_max_step_sigma,
                imm_mode_stickiness=args.state_space_imm_mode_stickiness,
                momentum_sigma_cm_sqrt_s=args.state_space_momentum_sigma_cm_sqrt_s,
                momentum_initial_sigma_cm_sqrt_s=args.state_space_momentum_initial_sigma_cm_sqrt_s,
                momentum_velocity_decay=args.state_space_momentum_velocity_decay,
                momentum_candidate_top_k=args.state_space_momentum_candidate_top_k,
            ),
        )

    def goal_state_space_model(name: str) -> GoalStateSpaceReplayModel:
        return GoalStateSpaceReplayModel(
            candidate_goals=_session_goal_candidates(session),
            transition_sigma_cm_sqrt_s=args.goal_state_space_transition_sigma_cm_sqrt_s,
            lateral_sigma_scale=float(
                getattr(
                    args,
                    "goal_state_space_lateral_sigma_scale",
                    DEFAULT_GOAL_LATERAL_SIGMA_SCALE,
                )
            ),
            diffusion_mixture_weight=float(
                getattr(
                    args,
                    "goal_state_space_diffusion_mixture_weight",
                    DEFAULT_GOAL_DIFFUSION_MIXTURE_WEIGHT,
                )
            ),
            drift_speed_cm_s=args.goal_state_space_drift_speed_cm_s,
            max_step_sigma=args.goal_state_space_max_step_sigma,
            reset_probability=float(getattr(args, "goal_state_space_reset_probability", 0.0)),
            reset_initial_position_prior_weight=float(
                getattr(
                    args,
                    "goal_state_space_reset_initial_position_prior_weight",
                    DEFAULT_GOAL_RESET_INITIAL_POSITION_PRIOR_WEIGHT,
                )
            ),
            component_switch_probability=_goal_component_switch_probability(name, args),
            initial_position_prior_direction_mode=str(
                getattr(
                    args,
                    "goal_state_space_initial_position_prior_direction_mode",
                    DEFAULT_GOAL_INITIAL_POSITION_PRIOR_DIRECTION_MODE,
                )
            ),
            terminal_goal_prior_sigma_cm=float(
                getattr(args, "goal_state_space_terminal_prior_sigma_cm", 0.0)
            ),
            terminal_goal_prior_weight=float(
                getattr(
                    args,
                    "goal_state_space_terminal_goal_prior_weight",
                    DEFAULT_GOAL_TERMINAL_PRIOR_WEIGHT,
                )
            ),
            initial_goal_prior_sigma_cm=float(
                getattr(args, "goal_state_space_initial_goal_prior_sigma_cm", 0.0)
            ),
            initial_goal_prior_weight=float(
                getattr(
                    args,
                    "goal_state_space_initial_goal_prior_weight",
                    DEFAULT_GOAL_INITIAL_PRIOR_WEIGHT,
                )
            ),
            toward_direction_prior_weight=_goal_toward_direction_prior_weight(name, args),
            reverse_terminal_position_prior_weight=float(
                getattr(
                    args,
                    "goal_state_space_reverse_terminal_position_prior_weight",
                    DEFAULT_GOAL_REVERSE_TERMINAL_POSITION_PRIOR_WEIGHT,
                )
            ),
            direction_mode=_goal_direction_mode(name),
            name=name,
        )

    available = {
        "random": RandomModel(),
        "stationary": StationaryModel(),
        "stationary-gaussian": CandidateKinematicModel(
            mode="stationary", top_k=args.candidate_top_k, stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm, momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay, mode_stickiness=args.mode_stickiness, name="stationary-gaussian"),
        "diffusion": CandidateKinematicModel(
            mode="diffusion", top_k=args.candidate_top_k, stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm, momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay, mode_stickiness=args.mode_stickiness, name="diffusion"),
        "momentum": CandidateKinematicModel(
            mode="momentum", top_k=args.candidate_top_k, stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm, momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay, mode_stickiness=args.mode_stickiness, name="momentum"),
        "imm": CandidateKinematicModel(
            mode="imm", top_k=args.candidate_top_k, stationary_sigma_cm=args.stationary_sigma_cm,
            diffusion_sigma_cm=args.diffusion_sigma_cm, momentum_sigma_cm=args.momentum_sigma_cm,
            velocity_decay=args.velocity_decay, mode_stickiness=args.mode_stickiness, name="imm"),
        "sorted-spike-state-space-stationary": state_space_model("stationary"),
        "sorted-spike-state-space-diffusion": state_space_model("diffusion"),
        "sorted-spike-state-space-fragmented": state_space_model("fragmented"),
        "sorted-spike-state-space-jump": state_space_model("jump"),
        "sorted-spike-state-space-goal": goal_state_space_model("sorted-spike-state-space-goal"),
        "sorted-spike-state-space-goal-bidirectional": goal_state_space_model(
            "sorted-spike-state-space-goal-bidirectional"
        ),
        "sorted-spike-state-space-goal-forward-biased": goal_state_space_model(
            "sorted-spike-state-space-goal-forward-biased"
        ),
        "sorted-spike-state-space-goal-forward-biased-switching": goal_state_space_model(
            "sorted-spike-state-space-goal-forward-biased-switching"
        ),
        "sorted-spike-state-space-goal-reverse-biased": goal_state_space_model(
            "sorted-spike-state-space-goal-reverse-biased"
        ),
        "sorted-spike-state-space-momentum": state_space_model("momentum"),
        "sorted-spike-state-space-imm": state_space_model("imm"),
        "state-space-goal": goal_state_space_model("state-space-goal"),
        "state-space-goal-bidirectional": goal_state_space_model(
            "state-space-goal-bidirectional"
        ),
        "state-space-goal-forward-biased": goal_state_space_model(
            "state-space-goal-forward-biased"
        ),
        "state-space-goal-forward-biased-switching": goal_state_space_model(
            "state-space-goal-forward-biased-switching"
        ),
        "state-space-goal-reverse-biased": goal_state_space_model(
            "state-space-goal-reverse-biased"
        ),
        "clusterless-state-space-stationary": clusterless_state_space_model("stationary"),
        "clusterless-state-space-diffusion": clusterless_state_space_model("diffusion"),
        "clusterless-state-space-fragmented": clusterless_state_space_model("fragmented"),
        "clusterless-state-space-jump": clusterless_state_space_model("jump"),
        "clusterless-state-space-momentum": clusterless_state_space_model("momentum"),
        "clusterless-state-space-imm": clusterless_state_space_model("imm"),
    }
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"unknown models: {missing}; available: {sorted(available)}")
    return {name: available[name] for name in dict.fromkeys(names)}


def _session_goal_candidates(session) -> np.ndarray | None:
    if session is None:
        return None
    wells = _session_goal_table(session)
    if wells.empty:
        return None
    return wells[["well_x", "well_y"]].to_numpy(dtype=float)


def _goal_direction_mode(model_name: str) -> str:
    if (
        model_name.endswith("-goal-bidirectional")
        or model_name.endswith("-goal-forward-biased")
        or model_name.endswith("-goal-forward-biased-switching")
        or model_name.endswith("-goal-reverse-biased")
    ):
        return "bidirectional"
    return "toward"


def _goal_toward_direction_prior_weight(model_name: str, args) -> float:
    if model_name.endswith("-goal-forward-biased-switching"):
        return DEFAULT_GOAL_FORWARD_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT
    if model_name.endswith("-goal-forward-biased"):
        return DEFAULT_GOAL_FORWARD_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT
    if model_name.endswith("-goal-reverse-biased"):
        return DEFAULT_GOAL_REVERSE_BIASED_TOWARD_DIRECTION_PRIOR_WEIGHT
    return float(
        getattr(
            args,
            "goal_state_space_toward_direction_prior_weight",
            DEFAULT_GOAL_TOWARD_DIRECTION_PRIOR_WEIGHT,
        )
    )


def _goal_component_switch_probability(model_name: str, args) -> float:
    if model_name.endswith("-goal-forward-biased-switching"):
        return DEFAULT_GOAL_SWITCHING_COMPONENT_SWITCH_PROBABILITY
    return float(
        getattr(
            args,
            "goal_state_space_component_switch_probability",
            DEFAULT_GOAL_COMPONENT_SWITCH_PROBABILITY,
        )
    )


def _session_goal_table(session) -> pd.DataFrame:
    wells = infer_well_locations(session)
    if wells.empty:
        return wells
    return wells.sort_values("well_id").reset_index(drop=True)


def _goal_prior_weights_for_event(args, session, event_id: int, n_goals: int) -> np.ndarray | None:
    active_weight = float(getattr(args, "goal_state_space_active_goal_prior_weight", 0.0))
    if active_weight < 0.0 or active_weight > 1.0:
        raise ValueError("--goal-state-space-active-goal-prior-weight must be in [0, 1]")
    if active_weight <= 0.0:
        return None
    wells = _session_goal_table(session)
    if wells.empty or len(wells) != n_goals:
        return None
    active_goal_id = active_goal_at_time(session, session.ripple(int(event_id)).peak)
    if active_goal_id is None:
        return None
    matches = np.flatnonzero(wells["well_id"].to_numpy(dtype=int) == int(active_goal_id))
    if matches.size != 1:
        return None
    if n_goals == 1:
        return np.ones(1, dtype=float)
    weights = np.full(n_goals, (1.0 - active_weight) / (n_goals - 1), dtype=float)
    weights[int(matches[0])] = active_weight
    return weights


def _initial_position_prior_weights_for_event(
    args,
    session,
    event_id: int,
    bin_centers: np.ndarray,
) -> np.ndarray | None:
    return _position_prior_weights_for_event(
        args,
        session,
        event_id,
        bin_centers,
        sigma_attr="goal_state_space_ripple_position_prior_sigma_cm",
        weight_attr="goal_state_space_ripple_position_prior_weight",
        sigma_flag="--goal-state-space-ripple-position-prior-sigma-cm",
        weight_flag="--goal-state-space-ripple-position-prior-weight",
    )


def _reverse_terminal_position_prior_weights_for_event(
    args,
    session,
    event_id: int,
    bin_centers: np.ndarray,
) -> np.ndarray | None:
    return _position_prior_weights_for_event(
        args,
        session,
        event_id,
        bin_centers,
        sigma_attr="goal_state_space_reverse_terminal_position_prior_sigma_cm",
        weight_attr="goal_state_space_reverse_terminal_position_prior_weight",
        sigma_flag="--goal-state-space-reverse-terminal-position-prior-sigma-cm",
        weight_flag="--goal-state-space-reverse-terminal-position-prior-weight",
    )


def _position_prior_weights_for_event(
    args,
    session,
    event_id: int,
    bin_centers: np.ndarray,
    *,
    sigma_attr: str,
    weight_attr: str,
    sigma_flag: str,
    weight_flag: str,
) -> np.ndarray | None:
    sigma_cm = float(getattr(args, sigma_attr, 0.0))
    prior_weight = float(getattr(args, weight_attr, 1.0))
    if sigma_cm < 0.0:
        raise ValueError(f"{sigma_flag} must be non-negative")
    if prior_weight < 0.0 or prior_weight > 1.0:
        raise ValueError(f"{weight_flag} must be in [0, 1]")
    if sigma_cm <= 0.0 or prior_weight <= 0.0:
        return None
    centers = np.asarray(bin_centers, dtype=float)
    position = _position_at_time(session, session.ripple(int(event_id)).peak, centers.shape[1])
    if position is None:
        return None
    delta = centers - position[None, :]
    distances2 = np.sum(delta * delta, axis=1)
    weights = np.exp(-0.5 * distances2 / (sigma_cm * sigma_cm))
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
        return None
    gaussian = weights / float(weights.sum())
    if prior_weight >= 1.0:
        return gaussian
    uniform = np.full(centers.shape[0], 1.0 / centers.shape[0], dtype=float)
    blended = (1.0 - prior_weight) * uniform + prior_weight * gaussian
    return blended / float(blended.sum())


def _position_at_time(session, time_s: float, position_dim: int) -> np.ndarray | None:
    position = np.asarray(session.position, dtype=float)
    if position.ndim != 2 or position.shape[1] < position_dim + 1 or position.shape[0] == 0:
        return None
    keep = np.all(np.isfinite(position[:, : position_dim + 1]), axis=1)
    if not np.any(keep):
        return None
    clean = position[keep]
    index = int(np.argmin(np.abs(clean[:, 0] - float(time_s))))
    return clean[index, 1 : position_dim + 1]


def _model_for_event(args, session, event_id: int, model: object, bin_centers: np.ndarray) -> object:
    if not isinstance(model, GoalStateSpaceReplayModel):
        return model
    candidate_goals = model.candidate_goals
    goal_prior = None
    if candidate_goals is None:
        goal_prior = None
    else:
        goal_prior = _goal_prior_weights_for_event(args, session, event_id, len(candidate_goals))
    initial_position_prior = _initial_position_prior_weights_for_event(
        args,
        session,
        event_id,
        bin_centers,
    )
    reverse_terminal_position_prior = _reverse_terminal_position_prior_weights_for_event(
        args,
        session,
        event_id,
        bin_centers,
    )
    if (
        goal_prior is None
        and initial_position_prior is None
        and reverse_terminal_position_prior is None
    ):
        return model
    return replace(
        model,
        goal_prior_weights=goal_prior,
        initial_position_prior_weights=initial_position_prior,
        reverse_terminal_position_prior_weights=reverse_terminal_position_prior,
    )


def _family(model: str) -> str:
    if model in _TRAJ:
        return "trajectory"
    if model in _NONTRAJ:
        return "nontrajectory"
    return "other"


def _clusterless_mark_config(args) -> ClusterlessMarkConfig:
    return ClusterlessMarkConfig(
        encoding=EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
        ),
        mark_smoothing_sigma_bins=args.clusterless_mark_smoothing_sigma_bins,
        mark_prior_count=args.clusterless_mark_prior_count,
        mark_variance_floor=args.clusterless_mark_variance_floor,
        rate_floor_hz=args.clusterless_rate_floor_hz,
    )


def _score(args) -> pd.DataFrame:
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
        ),
    )
    models = _models(args, session=session)
    has_clusterless = any(isinstance(model, ClusterlessStateSpaceReplayModel) for model in models.values())
    clusterless_encoding = None
    if has_clusterless:
        clusterless_encoding = fit_clusterless_mark_encoding(
            session, _clusterless_mark_config(args),
        )
    emissions_cfg = EmissionConfig(
        time_bin_s=args.time_bin_s,
        spike_rate_scale=args.spike_rate_scale,
    )
    rows: list[dict[str, object]] = []

    for event_id in event_ids:
        sorted_emissions = build_emissions(session, encoding, int(event_id), emissions_cfg)
        clusterless_emissions = (
            build_clusterless_mark_emissions(session, clusterless_encoding, int(event_id), emissions_cfg)
            if clusterless_encoding is not None
            else None
        )
        if sorted_emissions.n_time == 0:
            continue
        for name, model in models.items():
            start = time.perf_counter()
            use_clusterless = isinstance(model, ClusterlessStateSpaceReplayModel)
            emissions = clusterless_emissions if use_clusterless else sorted_emissions
            bin_centers = clusterless_encoding.bin_centers if use_clusterless and clusterless_encoding is not None else encoding.bin_centers
            event_model = _model_for_event(args, session, int(event_id), model, bin_centers)
            assert emissions is not None
            try:
                if isinstance(event_model, CandidateKinematicModel):
                    cand = event_model.candidate_indices(emissions)
                    result = event_model.score(emissions, bin_centers, candidate_indices=cand)
                else:
                    result = event_model.score(emissions, bin_centers)
                model_name = str(result.model_name)
                row = {
                    "status": "success", "session": session.session_id, "event_index": int(event_id),
                    "model": model_name, "requested_model": name, "model_family": _family(model_name),
                    "log_evidence": float(result.log_likelihood), "n_time": int(result.n_time),
                    "n_spikes": int(result.n_spikes), "runtime_s": float(time.perf_counter() - start),
                    "error": "", "bin_size_cm": float(args.bin_size_cm),
                    "smoothing_sigma_bins": float(args.smoothing_sigma_bins),
                    "min_speed_cm_s": float(args.min_speed_cm_s),
                    "time_bin_s": float(args.time_bin_s),
                    "spike_rate_scale": float(args.spike_rate_scale),
                    "clusterless_mark_smoothing_sigma_bins": float(args.clusterless_mark_smoothing_sigma_bins),
                    "clusterless_mark_prior_count": float(args.clusterless_mark_prior_count),
                    "clusterless_mark_variance_floor": float(args.clusterless_mark_variance_floor),
                    "clusterless_rate_floor_hz": float(args.clusterless_rate_floor_hz),
                    "goal_state_space_transition_sigma_cm_sqrt_s": float(args.goal_state_space_transition_sigma_cm_sqrt_s),
                    "goal_state_space_lateral_sigma_scale": float(
                        getattr(
                            args,
                            "goal_state_space_lateral_sigma_scale",
                            DEFAULT_GOAL_LATERAL_SIGMA_SCALE,
                        )
                    ),
                    "goal_state_space_diffusion_mixture_weight": float(
                        getattr(
                            args,
                            "goal_state_space_diffusion_mixture_weight",
                            DEFAULT_GOAL_DIFFUSION_MIXTURE_WEIGHT,
                        )
                    ),
                    "goal_state_space_drift_speed_cm_s": float(args.goal_state_space_drift_speed_cm_s),
                    "goal_state_space_max_step_sigma": float(args.goal_state_space_max_step_sigma),
                    "goal_state_space_reset_probability": float(
                        getattr(args, "goal_state_space_reset_probability", 0.0)
                    ),
                    "goal_state_space_reset_initial_position_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_reset_initial_position_prior_weight",
                            DEFAULT_GOAL_RESET_INITIAL_POSITION_PRIOR_WEIGHT,
                        )
                    ),
                    "goal_state_space_component_switch_probability": float(
                        getattr(
                            args,
                            "goal_state_space_component_switch_probability",
                            DEFAULT_GOAL_COMPONENT_SWITCH_PROBABILITY,
                        )
                    ),
                    "goal_state_space_initial_position_prior_direction_mode": str(
                        getattr(
                            args,
                            "goal_state_space_initial_position_prior_direction_mode",
                            DEFAULT_GOAL_INITIAL_POSITION_PRIOR_DIRECTION_MODE,
                        )
                    ),
                    "goal_state_space_terminal_prior_sigma_cm": float(
                        getattr(args, "goal_state_space_terminal_prior_sigma_cm", 0.0)
                    ),
                    "goal_state_space_terminal_goal_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_terminal_goal_prior_weight",
                            DEFAULT_GOAL_TERMINAL_PRIOR_WEIGHT,
                        )
                    ),
                    "goal_state_space_initial_goal_prior_sigma_cm": float(
                        getattr(args, "goal_state_space_initial_goal_prior_sigma_cm", 0.0)
                    ),
                    "goal_state_space_initial_goal_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_initial_goal_prior_weight",
                            DEFAULT_GOAL_INITIAL_PRIOR_WEIGHT,
                        )
                    ),
                    "goal_state_space_toward_direction_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_toward_direction_prior_weight",
                            DEFAULT_GOAL_TOWARD_DIRECTION_PRIOR_WEIGHT,
                        )
                    ),
                    "goal_state_space_active_goal_prior_weight": float(
                        getattr(args, "goal_state_space_active_goal_prior_weight", 0.0)
                    ),
                    "goal_state_space_ripple_position_prior_sigma_cm": float(
                        getattr(args, "goal_state_space_ripple_position_prior_sigma_cm", 0.0)
                    ),
                    "goal_state_space_ripple_position_prior_weight": float(
                        getattr(args, "goal_state_space_ripple_position_prior_weight", 1.0)
                    ),
                    "goal_state_space_reverse_terminal_position_prior_sigma_cm": float(
                        getattr(
                            args,
                            "goal_state_space_reverse_terminal_position_prior_sigma_cm",
                            0.0,
                        )
                    ),
                    "goal_state_space_reverse_terminal_position_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_reverse_terminal_position_prior_weight",
                            DEFAULT_GOAL_REVERSE_TERMINAL_POSITION_PRIOR_WEIGHT,
                        )
                    ),
                }
                if use_clusterless and clusterless_encoding is not None:
                    row.update({
                        "clusterless_mark_features": int(clusterless_encoding.n_features),
                        "clusterless_spike_mark_source": clusterless_encoding.spike_mark_source,
                    })
                row.update({f"diagnostic_{key}": value for key, value in result.diagnostics.items()})
                rows.append(row)
                print(f"Scored {session.session_id} event {event_id} with {name}", flush=True)
            except Exception as exc:
                rows.append({
                    "status": "failure", "session": session.session_id, "event_index": int(event_id),
                    "model": name, "requested_model": name, "model_family": _family(name), "log_evidence": np.nan,
                    "n_time": int(emissions.n_time), "n_spikes": int(emissions.n_spikes),
                    "runtime_s": float(time.perf_counter() - start), "error": f"{type(exc).__name__}: {exc}",
                    "bin_size_cm": float(args.bin_size_cm),
                    "smoothing_sigma_bins": float(args.smoothing_sigma_bins),
                    "min_speed_cm_s": float(args.min_speed_cm_s),
                    "time_bin_s": float(args.time_bin_s),
                    "spike_rate_scale": float(args.spike_rate_scale),
                    "clusterless_mark_smoothing_sigma_bins": float(args.clusterless_mark_smoothing_sigma_bins),
                    "clusterless_mark_prior_count": float(args.clusterless_mark_prior_count),
                    "clusterless_mark_variance_floor": float(args.clusterless_mark_variance_floor),
                    "clusterless_rate_floor_hz": float(args.clusterless_rate_floor_hz),
                    "goal_state_space_transition_sigma_cm_sqrt_s": float(args.goal_state_space_transition_sigma_cm_sqrt_s),
                    "goal_state_space_lateral_sigma_scale": float(
                        getattr(
                            args,
                            "goal_state_space_lateral_sigma_scale",
                            DEFAULT_GOAL_LATERAL_SIGMA_SCALE,
                        )
                    ),
                    "goal_state_space_diffusion_mixture_weight": float(
                        getattr(
                            args,
                            "goal_state_space_diffusion_mixture_weight",
                            DEFAULT_GOAL_DIFFUSION_MIXTURE_WEIGHT,
                        )
                    ),
                    "goal_state_space_drift_speed_cm_s": float(args.goal_state_space_drift_speed_cm_s),
                    "goal_state_space_max_step_sigma": float(args.goal_state_space_max_step_sigma),
                    "goal_state_space_reset_probability": float(
                        getattr(args, "goal_state_space_reset_probability", 0.0)
                    ),
                    "goal_state_space_reset_initial_position_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_reset_initial_position_prior_weight",
                            DEFAULT_GOAL_RESET_INITIAL_POSITION_PRIOR_WEIGHT,
                        )
                    ),
                    "goal_state_space_component_switch_probability": float(
                        getattr(
                            args,
                            "goal_state_space_component_switch_probability",
                            DEFAULT_GOAL_COMPONENT_SWITCH_PROBABILITY,
                        )
                    ),
                    "goal_state_space_initial_position_prior_direction_mode": str(
                        getattr(
                            args,
                            "goal_state_space_initial_position_prior_direction_mode",
                            DEFAULT_GOAL_INITIAL_POSITION_PRIOR_DIRECTION_MODE,
                        )
                    ),
                    "goal_state_space_terminal_prior_sigma_cm": float(
                        getattr(args, "goal_state_space_terminal_prior_sigma_cm", 0.0)
                    ),
                    "goal_state_space_terminal_goal_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_terminal_goal_prior_weight",
                            DEFAULT_GOAL_TERMINAL_PRIOR_WEIGHT,
                        )
                    ),
                    "goal_state_space_initial_goal_prior_sigma_cm": float(
                        getattr(args, "goal_state_space_initial_goal_prior_sigma_cm", 0.0)
                    ),
                    "goal_state_space_initial_goal_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_initial_goal_prior_weight",
                            DEFAULT_GOAL_INITIAL_PRIOR_WEIGHT,
                        )
                    ),
                    "goal_state_space_toward_direction_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_toward_direction_prior_weight",
                            DEFAULT_GOAL_TOWARD_DIRECTION_PRIOR_WEIGHT,
                        )
                    ),
                    "goal_state_space_active_goal_prior_weight": float(
                        getattr(args, "goal_state_space_active_goal_prior_weight", 0.0)
                    ),
                    "goal_state_space_ripple_position_prior_sigma_cm": float(
                        getattr(args, "goal_state_space_ripple_position_prior_sigma_cm", 0.0)
                    ),
                    "goal_state_space_ripple_position_prior_weight": float(
                        getattr(args, "goal_state_space_ripple_position_prior_weight", 1.0)
                    ),
                    "goal_state_space_reverse_terminal_position_prior_sigma_cm": float(
                        getattr(
                            args,
                            "goal_state_space_reverse_terminal_position_prior_sigma_cm",
                            0.0,
                        )
                    ),
                    "goal_state_space_reverse_terminal_position_prior_weight": float(
                        getattr(
                            args,
                            "goal_state_space_reverse_terminal_position_prior_weight",
                            DEFAULT_GOAL_REVERSE_TERMINAL_POSITION_PRIOR_WEIGHT,
                        )
                    ),
                })
                if not args.continue_on_error:
                    raise
    return _add_evidence_columns(pd.DataFrame(rows))


def _add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _ensure_evidence_support_columns(df)
    groups = []
    for _, g in df.groupby(["session", "event_index"], sort=False):
        g = g.copy()
        s = g[g["status"] == "success"]
        g["relative_log_evidence"] = np.nan
        g["model_probability"] = np.nan
        g["is_best_model"] = False
        g["best_model"] = ""
        g["best_trajectory_model"] = ""
        g["delta_vs_trajectory_best"] = np.nan
        g["best_nontrajectory_model"] = ""
        g["delta_vs_nontrajectory_best"] = np.nan
        g["truncated_relative_log_evidence"] = np.nan
        g["is_best_truncated_lower_bound"] = False
        g["best_truncated_lower_bound_model"] = ""
        if s.empty:
            groups.append(g)
            continue

        exact = s[s["evidence_comparable"].fillna(False).astype(bool)]
        if not exact.empty:
            vals = exact["log_evidence"].to_numpy(float)
            maxv = float(np.max(vals))
            probs = np.exp(vals - logsumexp(vals))
            best_index = exact.index[int(np.argmax(vals))]
            best = str(g.loc[best_index, "model"])
            g.loc[exact.index, "relative_log_evidence"] = vals - maxv
            g.loc[exact.index, "model_probability"] = probs
            g.loc[best_index, "is_best_model"] = True
            g["best_model"] = best

        for family, col in (("trajectory", "best_trajectory_model"), ("nontrajectory", "best_nontrajectory_model")):
            subset = exact[exact["model_family"] == family]
            if not subset.empty:
                bidx = int(np.argmax(subset["log_evidence"].to_numpy(float)))
                bname = str(subset.iloc[bidx]["model"])
                blog = float(subset.iloc[bidx]["log_evidence"])
                g[col] = bname
                g.loc[exact.index, f"delta_vs_{family}_best"] = g.loc[exact.index, "log_evidence"] - blog

        truncated = s[s["evidence_support"].eq(TRUNCATED_EVIDENCE_SUPPORT)]
        if not truncated.empty:
            lower_bounds = truncated["log_evidence"].to_numpy(float)
            max_lower_bound = float(np.max(lower_bounds))
            best_truncated_index = truncated.index[int(np.argmax(lower_bounds))]
            best_truncated = str(g.loc[best_truncated_index, "model"])
            g.loc[truncated.index, "truncated_relative_log_evidence"] = lower_bounds - max_lower_bound
            g.loc[best_truncated_index, "is_best_truncated_lower_bound"] = True
            g["best_truncated_lower_bound_model"] = best_truncated
        groups.append(g)
    return pd.concat(groups, ignore_index=True).sort_values(["event_index", "model"]).reset_index(drop=True)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_evidence_support_columns(df)
    ok = df[df["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    ok = ok.copy()
    if "is_best_truncated_lower_bound" not in ok:
        ok["is_best_truncated_lower_bound"] = False
    if "truncated_relative_log_evidence" not in ok:
        ok["truncated_relative_log_evidence"] = np.nan
    out = ok.groupby(["model", "model_family", "evidence_support", "evidence_comparable"], as_index=False).agg(
        events=("event_index", "count"), wins=("is_best_model", "sum"),
        truncated_lower_bound_wins=("is_best_truncated_lower_bound", "sum"),
        mean_log_evidence=("log_evidence", "mean"), median_log_evidence=("log_evidence", "median"),
        mean_relative_log_evidence=("relative_log_evidence", "mean"),
        median_relative_log_evidence=("relative_log_evidence", "median"),
        mean_model_probability=("model_probability", "mean"),
        median_model_probability=("model_probability", "median"),
        mean_truncated_relative_log_evidence=("truncated_relative_log_evidence", "mean"),
        median_truncated_relative_log_evidence=("truncated_relative_log_evidence", "median"),
        mean_runtime_s=("runtime_s", "mean"),
    )
    out["win_fraction"] = out["wins"] / out["events"].clip(lower=1)
    out["truncated_lower_bound_win_fraction"] = out["truncated_lower_bound_wins"] / out["events"].clip(lower=1)
    return out.sort_values(
        ["evidence_comparable", "wins", "truncated_lower_bound_wins", "mean_log_evidence"],
        ascending=[False, False, False, False],
    )


def _counts(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_evidence_support_columns(df)
    ok = df[df["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    base = ok.drop_duplicates(["session", "event_index"])
    rows = []
    for col in (
        "best_model",
        "best_trajectory_model",
        "best_nontrajectory_model",
        "best_truncated_lower_bound_model",
    ):
        if col not in base:
            continue
        values = base[col].dropna().astype(str)
        values = values[values != ""]
        if values.empty:
            continue
        vc = values.value_counts().rename_axis("model").reset_index(name="events")
        vc["comparison"] = col
        rows.extend(vc.to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=["comparison", "model", "events"])
    return pd.DataFrame(rows)[["comparison", "model", "events"]]


def _write(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "event_model_evidence.csv", index=False)
    _summary(df).to_csv(outdir / "model_evidence_summary.csv", index=False)
    _counts(df).to_csv(outdir / "best_model_counts.csv", index=False)
    ok = df[df["status"] == "success"]
    metrics = ["log_evidence", "relative_log_evidence", "model_probability"]
    if "truncated_relative_log_evidence" in ok:
        metrics.append("truncated_relative_log_evidence")
    for metric in metrics:
        ok.pivot_table(index=["session", "event_index"], columns="model", values=metric, aggfunc="first").reset_index().to_csv(outdir / f"event_model_pivot_{metric}.csv", index=False)


def main() -> int:
    p = argparse.ArgumentParser(description="Run a session-scoped replay model-evidence benchmark.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--events", default="0-25")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--models", default="random stationary stationary-gaussian diffusion momentum imm")
    p.add_argument("--candidate-top-k", type=int, default=64)
    p.add_argument("--stationary-sigma-cm", type=float, default=2.0)
    p.add_argument("--diffusion-sigma-cm", type=float, default=12.0)
    p.add_argument("--momentum-sigma-cm", type=float, default=12.0)
    p.add_argument("--velocity-decay", type=float, default=0.95)
    p.add_argument("--mode-stickiness", type=float, default=0.94)
    p.add_argument("--state-space-stationary-sigma-cm", type=float, default=2.0)
    p.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    p.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    p.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)
    p.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    p.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=85.0)
    p.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.95)
    p.add_argument("--state-space-momentum-candidate-top-k", type=int, default=128)
    p.add_argument(
        "--goal-state-space-transition-sigma-cm-sqrt-s",
        type=float,
        default=DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S,
    )
    p.add_argument(
        "--goal-state-space-drift-speed-cm-s",
        type=float,
        default=DEFAULT_GOAL_DRIFT_SPEED_CM_S,
    )
    p.add_argument(
        "--goal-state-space-lateral-sigma-scale",
        type=float,
        default=DEFAULT_GOAL_LATERAL_SIGMA_SCALE,
        help=(
            "Scale applied to goal-state-space transition noise perpendicular "
            "to the source-goal axis; 1 keeps isotropic Gaussian transitions."
        ),
    )
    p.add_argument(
        "--goal-state-space-diffusion-mixture-weight",
        type=float,
        default=DEFAULT_GOAL_DIFFUSION_MIXTURE_WEIGHT,
        help=(
            "Mixture weight in [0, 1] for adding a zero-drift diffusion "
            "transition to each goal-directed transition."
        ),
    )
    p.add_argument(
        "--goal-state-space-max-step-sigma",
        type=float,
        default=DEFAULT_GOAL_MAX_STEP_SIGMA,
    )
    p.add_argument(
        "--goal-state-space-reset-probability",
        type=float,
        default=DEFAULT_GOAL_RESET_PROBABILITY,
        help=(
            "Per-time-bin probability that goal-state-space dynamics reset "
            "position to the uniform spatial prior while keeping the latent goal fixed."
        ),
    )
    p.add_argument(
        "--goal-state-space-reset-initial-position-prior-weight",
        type=float,
        default=DEFAULT_GOAL_RESET_INITIAL_POSITION_PRIOR_WEIGHT,
        help=(
            "Blend weight in [0, 1] for drawing goal-state-space reset "
            "positions from the event initial-position prior instead of uniform space."
        ),
    )
    p.add_argument(
        "--goal-state-space-component-switch-probability",
        type=float,
        default=DEFAULT_GOAL_COMPONENT_SWITCH_PROBABILITY,
        help=(
            "Per-transition probability of redrawing the latent goal/direction "
            "component from its prior while preserving predicted position."
        ),
    )
    p.add_argument(
        "--goal-state-space-initial-position-prior-direction-mode",
        choices=("all", "toward", "away"),
        default=DEFAULT_GOAL_INITIAL_POSITION_PRIOR_DIRECTION_MODE,
        help=(
            "Which goal-state-space direction components receive the "
            "event initial-position prior; 'all' preserves the original "
            "component-agnostic prior."
        ),
    )
    p.add_argument(
        "--goal-state-space-terminal-prior-sigma-cm",
        type=float,
        default=DEFAULT_GOAL_TERMINAL_PRIOR_SIGMA_CM,
        help=(
            "If positive, apply a mean-one terminal likelihood factor centered "
            "on each candidate goal for toward-goal components."
        ),
    )
    p.add_argument(
        "--goal-state-space-terminal-goal-prior-weight",
        type=float,
        default=DEFAULT_GOAL_TERMINAL_PRIOR_WEIGHT,
        help=(
            "Blend weight in [0, 1] for the terminal goal prior factor; "
            "1 uses the full mean-one factor and 0 disables it."
        ),
    )
    p.add_argument(
        "--goal-state-space-initial-goal-prior-sigma-cm",
        type=float,
        default=DEFAULT_GOAL_INITIAL_PRIOR_SIGMA_CM,
        help=(
            "If positive, apply a mean-one initial likelihood factor centered "
            "on each candidate goal for away-from-goal components."
        ),
    )
    p.add_argument(
        "--goal-state-space-initial-goal-prior-weight",
        type=float,
        default=DEFAULT_GOAL_INITIAL_PRIOR_WEIGHT,
        help=(
            "Blend weight in [0, 1] for the initial goal prior factor; "
            "1 uses the full mean-one factor and 0 disables it."
        ),
    )
    p.add_argument(
        "--goal-state-space-toward-direction-prior-weight",
        type=float,
        default=DEFAULT_GOAL_TOWARD_DIRECTION_PRIOR_WEIGHT,
        help=(
            "For bidirectional goal-state-space models, prior probability "
            "assigned to toward-goal components within each candidate goal."
        ),
    )
    p.add_argument(
        "--goal-state-space-active-goal-prior-weight",
        type=float,
        default=0.0,
        help=(
            "If positive, assign this prior probability to the well active at "
            "the ripple peak for goal-state-space models and spread the "
            "remaining mass uniformly across other inferred wells."
        ),
    )
    p.add_argument(
        "--goal-state-space-ripple-position-prior-sigma-cm",
        type=float,
        default=0.0,
        help=(
            "If positive, initialize goal-state-space models from a Gaussian "
            "position prior centered on the animal's position at ripple peak."
        ),
    )
    p.add_argument(
        "--goal-state-space-ripple-position-prior-weight",
        type=float,
        default=1.0,
        help=(
            "Mixture weight in [0, 1] for the ripple-position initial prior; "
            "values below 1 blend it with the uniform initial spatial prior."
        ),
    )
    p.add_argument(
        "--goal-state-space-reverse-terminal-position-prior-sigma-cm",
        type=float,
        default=0.0,
        help=(
            "If positive, apply a terminal position prior centered on the "
            "animal's ripple-peak position to away/reverse goal components."
        ),
    )
    p.add_argument(
        "--goal-state-space-reverse-terminal-position-prior-weight",
        type=float,
        default=DEFAULT_GOAL_REVERSE_TERMINAL_POSITION_PRIOR_WEIGHT,
        help=(
            "Mixture weight in [0, 1] for the reverse terminal position prior; "
            "1 uses the full mean-one prior factor and 0 disables it."
        ),
    )
    p.add_argument("--clusterless-mark-smoothing-sigma-bins", type=float, default=1.0)
    p.add_argument("--clusterless-mark-prior-count", type=float, default=1.0)
    p.add_argument("--clusterless-mark-variance-floor", type=float, default=1.0)
    p.add_argument("--clusterless-rate-floor-hz", type=float, default=1e-4)
    p.add_argument("--time-bin-s", type=float, default=0.02)
    p.add_argument(
        "--spike-rate-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to Poisson place-field rates during ripple scoring.",
    )
    p.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    p.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    p.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    p.add_argument("--output", default="results/model-evidence")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()
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
