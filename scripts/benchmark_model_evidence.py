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
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns as _ensure_evidence_support_columns,
)
from hipporeplayimm.accuracy_upgrades import (
    bootstrap_model_win_probabilities,
    model_probability_diagnostics,
)
from hipporeplayimm.goal_state_space import GoalStateSpaceReplayModel
from hipporeplayimm.goal_state_space_integration import (
    DEFAULT_GOAL_DRIFT_SPEED_CM_S,
    DEFAULT_GOAL_MAX_STEP_SIGMA,
    DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S,
    GOAL_STATE_SPACE_MODEL_NAMES,
)
from hipporeplayimm.ground_truth import infer_well_locations
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
    "sorted-spike-state-space-momentum",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-imm",
    "sorted-spike-state-space-goal",
    "state-space-goal",
    "clusterless-state-space-diffusion",
    "clusterless-state-space-fragmented",
    "clusterless-state-space-jump",
    "clusterless-state-space-momentum",
    "clusterless-state-space-first-order-imm",
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
        if name == "clusterless-state-space-first-order-imm":
            continue
        if name:
            names.append(name)
    if not names:
        raise ValueError("no models selected")

    def state_space_config(mode: str) -> StateSpaceDecoderConfig:
        return StateSpaceDecoderConfig(
            mode=mode,
            stationary_sigma_cm=args.state_space_stationary_sigma_cm,
            diffusion_sigma_cm_sqrt_s=args.state_space_diffusion_sigma_cm_sqrt_s,
            max_step_sigma=args.state_space_max_step_sigma,
            imm_mode_stickiness=_state_space_mode_stickiness(args),
            momentum_sigma_cm_sqrt_s=args.state_space_momentum_sigma_cm_sqrt_s,
            momentum_initial_sigma_cm_sqrt_s=args.state_space_momentum_initial_sigma_cm_sqrt_s,
            momentum_velocity_decay=args.state_space_momentum_velocity_decay,
            momentum_candidate_top_k=args.state_space_momentum_candidate_top_k,
            momentum_candidate_mass_threshold=getattr(args, "state_space_momentum_candidate_mass_threshold", None),
            momentum_candidate_min_k=getattr(args, "state_space_momentum_candidate_min_k", 1),
            momentum_candidate_max_k=getattr(args, "state_space_momentum_candidate_max_k", 0),
            momentum_predicted_candidate_top_k=getattr(args, "state_space_momentum_predicted_candidate_top_k", 8),
        )

    def state_space_model(mode: str) -> SortedSpikeStateSpaceReplayModel:
        return SortedSpikeStateSpaceReplayModel(mode=mode, config=state_space_config(mode))

    def clusterless_state_space_model(mode: str) -> ClusterlessStateSpaceReplayModel:
        return ClusterlessStateSpaceReplayModel(
            mode=mode,
            config=state_space_config(mode),
            mark_likelihood=getattr(args, "clusterless_mark_likelihood", "local-kde"),
        )

    wants_goal_state_space = any(name in GOAL_STATE_SPACE_MODEL_NAMES for name in names)
    goal_candidates = _session_goal_candidates(session) if wants_goal_state_space else None

    def goal_state_space_model(name: str) -> GoalStateSpaceReplayModel:
        return GoalStateSpaceReplayModel(
            candidate_goals=goal_candidates,
            transition_sigma_cm_sqrt_s=getattr(args, "goal_state_space_transition_sigma_cm_sqrt_s", DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S),
            drift_speed_cm_s=getattr(args, "goal_state_space_drift_speed_cm_s", DEFAULT_GOAL_DRIFT_SPEED_CM_S),
            max_step_sigma=getattr(args, "goal_state_space_max_step_sigma", DEFAULT_GOAL_MAX_STEP_SIGMA),
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
        "sorted-spike-state-space-momentum": state_space_model("momentum"),
        "sorted-spike-state-space-first-order-imm": state_space_model("first-order-imm"),
        "sorted-spike-state-space-imm": state_space_model("imm"),
        "sorted-spike-state-space-goal": goal_state_space_model("sorted-spike-state-space-goal"),
        "state-space-goal": goal_state_space_model("state-space-goal"),
        "clusterless-state-space-stationary": clusterless_state_space_model("stationary"),
        "clusterless-state-space-diffusion": clusterless_state_space_model("diffusion"),
        "clusterless-state-space-fragmented": clusterless_state_space_model("fragmented"),
        "clusterless-state-space-jump": clusterless_state_space_model("jump"),
        "clusterless-state-space-momentum": clusterless_state_space_model("momentum"),
        "clusterless-state-space-first-order-imm": clusterless_state_space_model("first-order-imm"),
        "clusterless-state-space-imm": clusterless_state_space_model("imm"),
    }
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"unknown models: {missing}; available: {sorted(available)}")
    return {name: available[name] for name in dict.fromkeys(names)}


def _state_space_mode_stickiness(args) -> float:
    tau_s = float(getattr(args, "state_space_imm_switch_tau_s", 0.0))
    if tau_s <= 0.0:
        return float(args.state_space_imm_mode_stickiness)
    return float(np.exp(-float(getattr(args, "time_bin_s", 0.02)) / tau_s))


def _family(model: str) -> str:
    if model in _TRAJ:
        return "trajectory"
    if model in _NONTRAJ:
        return "nontrajectory"
    return "other"


def _session_goal_candidates(session) -> np.ndarray | None:
    if session is None:
        return None
    wells = infer_well_locations(session)
    if wells.empty:
        return None
    return wells[["well_x", "well_y"]].to_numpy(dtype=float)


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
        mark_likelihood=args.clusterless_mark_likelihood,
        mark_kde_bandwidth=args.clusterless_mark_kde_bandwidth,
        mark_kde_spatial_sigma_bins=args.clusterless_mark_kde_spatial_sigma_bins,
        mark_kde_max_neighbors=args.clusterless_mark_kde_max_neighbors,
    )


def _optional_float_setting(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _optional_float_argument(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


def _state_space_metadata(args) -> dict[str, object]:
    return {
        "state_space_momentum_predicted_candidate_top_k": int(getattr(args, "state_space_momentum_predicted_candidate_top_k", 8)),
        "state_space_imm_switch_tau_s": float(getattr(args, "state_space_imm_switch_tau_s", 0.0)),
        "state_space_effective_imm_mode_stickiness": float(_state_space_mode_stickiness(args)),
        "goal_state_space_transition_sigma_cm_sqrt_s": float(getattr(args, "goal_state_space_transition_sigma_cm_sqrt_s", DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S)),
        "goal_state_space_drift_speed_cm_s": float(getattr(args, "goal_state_space_drift_speed_cm_s", DEFAULT_GOAL_DRIFT_SPEED_CM_S)),
        "goal_state_space_max_step_sigma": float(getattr(args, "goal_state_space_max_step_sigma", DEFAULT_GOAL_MAX_STEP_SIGMA)),
    }


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
        likelihood_temperature=args.emission_likelihood_temperature,
        negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
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
            assert emissions is not None
            try:
                if isinstance(model, CandidateKinematicModel):
                    cand = model.candidate_indices(emissions)
                    result = model.score(emissions, bin_centers, candidate_indices=cand)
                else:
                    result = model.score(emissions, bin_centers)
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
                    "emission_likelihood_temperature": float(args.emission_likelihood_temperature),
                    "emission_negative_binomial_overdispersion": float(args.emission_negative_binomial_overdispersion),
                    "clusterless_mark_smoothing_sigma_bins": float(args.clusterless_mark_smoothing_sigma_bins),
                    "clusterless_mark_prior_count": float(args.clusterless_mark_prior_count),
                    "clusterless_mark_variance_floor": float(args.clusterless_mark_variance_floor),
                    "clusterless_rate_floor_hz": float(args.clusterless_rate_floor_hz),
                    "clusterless_mark_likelihood": str(args.clusterless_mark_likelihood),
                    "clusterless_mark_kde_bandwidth": _optional_float_setting(args.clusterless_mark_kde_bandwidth),
                    "clusterless_mark_kde_spatial_sigma_bins": _optional_float_setting(args.clusterless_mark_kde_spatial_sigma_bins),
                    "clusterless_mark_kde_max_neighbors": int(args.clusterless_mark_kde_max_neighbors),
                    **_state_space_metadata(args),
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
                    "emission_likelihood_temperature": float(args.emission_likelihood_temperature),
                    "emission_negative_binomial_overdispersion": float(args.emission_negative_binomial_overdispersion),
                    "clusterless_mark_smoothing_sigma_bins": float(args.clusterless_mark_smoothing_sigma_bins),
                    "clusterless_mark_prior_count": float(args.clusterless_mark_prior_count),
                    "clusterless_mark_variance_floor": float(args.clusterless_mark_variance_floor),
                    "clusterless_rate_floor_hz": float(args.clusterless_rate_floor_hz),
                    "clusterless_mark_likelihood": str(args.clusterless_mark_likelihood),
                    "clusterless_mark_kde_bandwidth": _optional_float_setting(args.clusterless_mark_kde_bandwidth),
                    "clusterless_mark_kde_spatial_sigma_bins": _optional_float_setting(args.clusterless_mark_kde_spatial_sigma_bins),
                    "clusterless_mark_kde_max_neighbors": int(args.clusterless_mark_kde_max_neighbors),
                    **_state_space_metadata(args),
                })
                if not args.continue_on_error:
                    raise
    return _add_evidence_columns(pd.DataFrame(rows))


def _event_group_columns(df: pd.DataFrame) -> list[str]:
    """Columns that identify one comparable model-choice unit."""

    columns = ["session", "event_index"]
    for optional in ("window_index", "benchmark_cell_split_index"):
        if optional in df.columns:
            columns.append(optional)
    return columns


def _add_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _ensure_evidence_support_columns(df)
    groups = []
    group_columns = _event_group_columns(df)
    for _, g in df.groupby(group_columns, sort=False):
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
    out = pd.concat(groups, ignore_index=True)
    sort_columns = [column for column in (*group_columns, "model") if column in out.columns]
    return out.sort_values(sort_columns).reset_index(drop=True)


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


def _summary_for_support(summary: pd.DataFrame, support: str) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    return summary[summary["evidence_support"].eq(support)].copy()


def _support_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_evidence_support_columns(df)
    if df.empty:
        return pd.DataFrame(columns=["evidence_support", "evidence_comparison", "evidence_comparable", "rows", "successful_rows", "events", "models"])
    return (
        df.groupby(["evidence_support", "evidence_comparison", "evidence_comparable"], dropna=False)
        .agg(
            rows=("model", "size"),
            successful_rows=("status", lambda status: int(status.eq("success").sum())),
            events=("event_index", "nunique"),
            models=("model", "nunique"),
        )
        .reset_index()
        .sort_values(["evidence_comparable", "evidence_support"], ascending=[False, True])
    )


def _counts(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_evidence_support_columns(df)
    ok = df[df["status"] == "success"]
    if ok.empty:
        return pd.DataFrame()
    base = ok.drop_duplicates(_event_group_columns(ok))
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
    summary = _summary(df)
    summary.to_csv(outdir / "model_evidence_summary.csv", index=False)
    _summary_for_support(summary, EXACT_EVIDENCE_SUPPORT).to_csv(outdir / "exact_model_evidence_summary.csv", index=False)
    _summary_for_support(summary, TRUNCATED_EVIDENCE_SUPPORT).to_csv(outdir / "truncated_lower_bound_summary.csv", index=False)
    _counts(df).to_csv(outdir / "best_model_counts.csv", index=False)
    _support_counts(df).to_csv(outdir / "evidence_support_counts.csv", index=False)
    ok = df[df["status"] == "success"]
    group_columns = _event_group_columns(df)
    diagnostics = model_probability_diagnostics(ok, group_columns=group_columns)
    diagnostics.to_csv(outdir / "model_probability_diagnostics.csv", index=False)
    bootstrap = bootstrap_model_win_probabilities(
        ok,
        group_columns=group_columns,
        n_bootstrap=1000,
        random_seed=1,
    )
    bootstrap.to_csv(outdir / "bootstrap_model_win_probabilities.csv", index=False)
    metrics = ["log_evidence", "relative_log_evidence", "model_probability"]
    if "truncated_relative_log_evidence" in ok:
        metrics.append("truncated_relative_log_evidence")
    for metric in metrics:
        ok.pivot_table(index=group_columns, columns="model", values=metric, aggfunc="first").reset_index().to_csv(
            outdir / f"event_model_pivot_{metric}.csv",
            index=False,
        )


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
    p.add_argument("--state-space-imm-switch-tau-s", type=float, default=0.0)
    p.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    p.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=85.0)
    p.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.95)
    p.add_argument("--state-space-momentum-candidate-top-k", type=int, default=128)
    p.add_argument("--state-space-momentum-predicted-candidate-top-k", type=int, default=8)
    p.add_argument(
        "--state-space-momentum-candidate-mass-threshold",
        type=float,
        default=None,
        help="Enable adaptive candidate support retaining this normalized emission mass.",
    )
    p.add_argument(
        "--state-space-momentum-candidate-min-k",
        type=int,
        default=1,
        help="Minimum per-bin support when adaptive candidate support is enabled.",
    )
    p.add_argument(
        "--state-space-momentum-candidate-max-k",
        type=int,
        default=0,
        help="Maximum adaptive per-bin support; 0 means unbounded.",
    )
    p.add_argument("--clusterless-mark-smoothing-sigma-bins", type=float, default=1.0)
    p.add_argument("--clusterless-mark-prior-count", type=float, default=1.0)
    p.add_argument("--clusterless-mark-variance-floor", type=float, default=1.0)
    p.add_argument("--clusterless-rate-floor-hz", type=float, default=1e-4)
    p.add_argument(
        "--clusterless-mark-likelihood",
        default="local-kde",
        help="Clusterless mark likelihood: local-kde or diagonal-gaussian. Aliases accepted by ClusterlessMarkConfig are also valid.",
    )
    p.add_argument(
        "--clusterless-mark-kde-bandwidth",
        type=_optional_float_argument,
        default=None,
        help="Optional scalar mark-space KDE bandwidth. Empty/default uses the data-adaptive bandwidth.",
    )
    p.add_argument(
        "--clusterless-mark-kde-spatial-sigma-bins",
        type=_optional_float_argument,
        default=None,
        help="Optional spatial weighting sigma, in grid bins, for local mark KDE support. Empty/default reuses clusterless mark smoothing sigma.",
    )
    p.add_argument("--clusterless-mark-kde-max-neighbors", type=int, default=256)
    p.add_argument("--goal-state-space-transition-sigma-cm-sqrt-s", type=float, default=DEFAULT_GOAL_TRANSITION_SIGMA_CM_SQRT_S)
    p.add_argument("--goal-state-space-drift-speed-cm-s", type=float, default=DEFAULT_GOAL_DRIFT_SPEED_CM_S)
    p.add_argument("--goal-state-space-max-step-sigma", type=float, default=DEFAULT_GOAL_MAX_STEP_SIGMA)
    p.add_argument("--time-bin-s", type=float, default=0.003)
    p.add_argument(
        "--spike-rate-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to Poisson place-field rates during ripple scoring.",
    )
    p.add_argument(
        "--emission-likelihood-temperature",
        type=float,
        default=1.0,
        help="Divide emission log likelihoods by this positive temperature; values >1 flatten the emission model.",
    )
    p.add_argument(
        "--emission-negative-binomial-overdispersion",
        type=float,
        default=0.0,
        help="Use a negative-binomial sorted-spike count model with variance mean + alpha * mean**2; 0 keeps the Poisson model.",
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
    summary = _summary(df)
    print(summary.to_string(index=False))
    print("\nBest-model counts:")
    print(_counts(df).to_string(index=False))
    support_counts = _support_counts(df)
    if not support_counts.empty:
        print("\nEvidence-support counts:")
        print(support_counts.to_string(index=False))
        print(
            "\nInterpretation: exact_full_grid rows are comparable model evidences; "
            "truncated_full_grid rows are candidate-support lower bounds and must be ranked only within the lower-bound diagnostic group."
        )
    print(f"\nRows: {len(df)}")
    print(f"Failures: {int((df['status'] != 'success').sum())}")
    _write(df, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
