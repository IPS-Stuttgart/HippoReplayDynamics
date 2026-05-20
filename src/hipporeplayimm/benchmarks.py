"""Open-field held-out likelihood benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .clusterless import (
    ClusterlessMarkConfig,
    ClusterlessStateSpaceReplayModel,
    _normalize_mark_likelihood,
    build_clusterless_mark_emissions,
    fit_clusterless_mark_encoding,
)
from .data import ReplaySession, SpikeMarkData, load_open_field_sessions
from .encoding import EmissionConfig, EncodingConfig, build_emissions, fit_place_field_encoding
from .evidence_reporting import (
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)
from .models import CandidateKinematicModel, RandomModel, StationaryModel
from .pyrecest_models import PyRecEstGoalParticleIMMModel, PyRecEstGoalParticleModel
from .sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from .state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


@dataclass(frozen=True)
class BenchmarkConfig:
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    emissions: EmissionConfig = field(default_factory=EmissionConfig)
    test_cell_fraction: float = 0.25
    max_events_per_session: int | None = None
    candidate_top_k: int = 64
    clusterless_mark_smoothing_sigma_bins: float = 1.0
    clusterless_mark_prior_count: float = 1.0
    clusterless_mark_variance_floor: float = 1.0
    clusterless_rate_floor_hz: float = 1e-4
    clusterless_mark_likelihood: str = "local-kde"
    clusterless_mark_kde_bandwidth: float | None = None
    clusterless_mark_kde_spatial_sigma_bins: float | None = None
    clusterless_mark_kde_max_neighbors: int = 256
    clusterless_mark_group_by: str = "auto"
    pyrecest_particles: int = 512
    pyrecest_alpha: float = 0.80
    pyrecest_beta: float = 1.00
    pyrecest_process_noise_sigma_cm_s: float = 60.0
    pyrecest_position_jump_sigma_cm: float = 25.0
    pyrecest_jump_probability: float = 0.03
    pyrecest_goal_reset_probability: float = 0.02
    pyrecest_position_proposal_probability: float = 0.0
    pyrecest_initial_velocity_sigma_cm_s: float = 120.0
    pyrecest_imm_mode_stickiness: float = 0.95
    pyrecest_imm_stationary_velocity_decay: float = 0.0
    pyrecest_imm_diffusion_velocity_decay: float = 0.0
    pyrecest_imm_momentum_velocity_decay: float = 0.95
    pyrecest_imm_jump_fraction: float = 0.9
    pyrecest_imm_jump_velocity_decay: float = 0.25
    state_space_valid_occupancy_threshold_s: float = 0.0
    random_seed: int = 1
    event_epoch: str = "run"
    models: tuple[str, ...] = ("random", "stationary", "diffusion", "momentum", "imm")


@dataclass
class BenchmarkResult:
    rows: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        if self.rows.empty:
            return pd.DataFrame()
        rows = ensure_evidence_support_columns(self.rows)
        metric_defaults = {
            "delta_vs_best_static": np.nan,
            "bits_per_spike_vs_best_static": np.nan,
            "lower_bound_delta_vs_best_static": np.nan,
            "lower_bound_bits_per_spike_vs_best_static": np.nan,
            "delta_vs_best_static_truncated_lower_bound": np.nan,
            "bits_per_spike_vs_best_static_truncated_lower_bound": np.nan,
        }
        for column, default in metric_defaults.items():
            if column not in rows:
                rows[column] = default
        grouped = rows.groupby(["model", "evidence_support", "evidence_comparable"], as_index=False)
        return grouped.agg(
            events=("heldout_log_likelihood", "count"),
            mean_heldout_log_likelihood=("heldout_log_likelihood", "mean"),
            mean_delta_vs_best_static=("delta_vs_best_static", "mean"),
            mean_bits_per_spike_vs_best_static=("bits_per_spike_vs_best_static", "mean"),
            mean_lower_bound_delta_vs_best_static=("lower_bound_delta_vs_best_static", "mean"),
            mean_lower_bound_bits_per_spike_vs_best_static=("lower_bound_bits_per_spike_vs_best_static", "mean"),
            mean_delta_vs_best_static_truncated_lower_bound=("delta_vs_best_static_truncated_lower_bound", "mean"),
            mean_bits_per_spike_vs_best_static_truncated_lower_bound=("bits_per_spike_vs_best_static_truncated_lower_bound", "mean"),
        )

    def to_csv(self, path: str | Path) -> None:
        self.rows.to_csv(path, index=False)


def run_open_field_benchmark(root: str | Path, config: BenchmarkConfig | None = None) -> BenchmarkResult:
    """Run held-out open-field benchmark across Rat1-4 Open1-2 sessions."""

    config = BenchmarkConfig() if config is None else config
    sessions = load_open_field_sessions(root)
    rows = []
    for session in sessions:
        rows.extend(_score_session(session, config))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = _add_relative_metrics(frame)
    return BenchmarkResult(frame)


def bootstrap_delta_ci(
    rows: pd.DataFrame,
    model: str = "imm",
    value_column: str = "delta_vs_best_static",
    n_bootstrap: int = 1000,
    random_seed: int = 1,
) -> tuple[float, float]:
    """Paired bootstrap CI over event-level deltas."""

    values = rows.loc[rows["model"] == model, value_column].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_seed)
    means = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = float(np.mean(sample))
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def _score_session(session: ReplaySession, config: BenchmarkConfig) -> list[dict[str, object]]:
    encoding = fit_place_field_encoding(session, config.encoding)
    model_objects = _build_models(config, session=session)
    train_cells, test_cells = _split_cells(encoding.cell_ids, config.test_cell_fraction, config.random_seed)
    if test_cells.size == 0 or train_cells.size == 0:
        return []
    train_encoding = encoding.select_cells(train_cells)
    joint_encoding = encoding.select_cells(np.concatenate([train_cells, test_cells]))

    has_clusterless_models = any(_is_clusterless_model(model) for model in model_objects.values())
    clusterless_train_session: ReplaySession | None = None
    clusterless_joint_session: ReplaySession | None = None
    clusterless_joint_encoding = None
    if has_clusterless_models:
        clusterless_train_session = _session_with_mark_cell_subset(
            session,
            train_cells,
            role="train",
        )
        clusterless_joint_session = _session_with_mark_cell_subset(
            session,
            np.concatenate([train_cells, test_cells]),
            role="joint",
        )
        clusterless_config = _clusterless_mark_config(config)
        # Use one joint observation model for both terms in
        # log p(train + test) - log p(train), so the train contribution cancels
        # under identical clusterless rate and mark-likelihood parameters.
        clusterless_joint_encoding = fit_clusterless_mark_encoding(
            clusterless_joint_session,
            clusterless_config,
        )

    event_indices = _event_indices(session, config)
    rows: list[dict[str, object]] = []
    for event_index in event_indices:
        train_emissions = build_emissions(session, train_encoding, int(event_index), config.emissions)
        joint_emissions = build_emissions(session, joint_encoding, int(event_index), config.emissions)
        if train_emissions.n_time == 0 or joint_emissions.n_time == 0:
            continue
        clusterless_train_emissions = None
        clusterless_joint_emissions = None
        if has_clusterless_models:
            assert clusterless_train_session is not None
            assert clusterless_joint_session is not None
            assert clusterless_joint_encoding is not None
            clusterless_train_emissions = build_clusterless_mark_emissions(
                clusterless_train_session,
                clusterless_joint_encoding,
                int(event_index),
                config.emissions,
            )
            clusterless_joint_emissions = build_clusterless_mark_emissions(
                clusterless_joint_session,
                clusterless_joint_encoding,
                int(event_index),
                config.emissions,
            )
        for model_name, model in model_objects.items():
            if _is_clusterless_model(model):
                assert clusterless_train_emissions is not None
                assert clusterless_joint_emissions is not None
                assert clusterless_joint_encoding is not None
                model_train_emissions = clusterless_train_emissions
                model_joint_emissions = clusterless_joint_emissions
                model_bin_centers = clusterless_joint_encoding.bin_centers
                model_occupancy_s = clusterless_joint_encoding.occupancy_s
            else:
                model_train_emissions = train_emissions
                model_joint_emissions = joint_emissions
                model_bin_centers = encoding.bin_centers
                model_occupancy_s = encoding.occupancy_s
            train_score, joint_score = _score_train_joint_model(
                model,
                model_train_emissions,
                model_joint_emissions,
                model_bin_centers,
                occupancy_s=model_occupancy_s,
            )
            heldout = joint_score.log_likelihood - train_score.log_likelihood
            rows.append(
                {
                    "session": session.session_id,
                    "event_index": int(event_index),
                    "model": joint_score.model_name,
                    "requested_model": model_name,
                    "heldout_log_likelihood": float(heldout),
                    "joint_log_likelihood": float(joint_score.log_likelihood),
                    "train_log_likelihood": float(train_score.log_likelihood),
                    "test_spikes": int(model_joint_emissions.n_spikes - model_train_emissions.n_spikes),
                    "n_time": int(model_train_emissions.n_time),
                    "train_cell_ids": _format_cell_ids(train_cells),
                    "test_cell_ids": _format_cell_ids(test_cells),
                    **_benchmark_config_metadata(config),
                    **_session_mark_diagnostics(session),
                    **{
                        f"diagnostic_{key}": value
                        for key, value in joint_score.diagnostics.items()
                    },
                }
            )
    return rows


def _score_train_joint_model(model, train_emissions, joint_emissions, bin_centers, occupancy_s=None):
    if isinstance(model, StateSpaceReplayModel):
        candidates = _candidate_indices_for_model(model, train_emissions, bin_centers)
        train_score = model.score(
            train_emissions,
            bin_centers,
            candidate_indices=candidates,
            occupancy_s=occupancy_s,
        )
        joint_score = model.score(
            joint_emissions,
            bin_centers,
            candidate_indices=candidates,
            occupancy_s=occupancy_s,
        )
        return train_score, joint_score
    if hasattr(model, "candidate_indices"):
        candidates = _candidate_indices_for_model(model, train_emissions, bin_centers)
        train_score = model.score(train_emissions, bin_centers, candidate_indices=candidates)
        joint_score = model.score(joint_emissions, bin_centers, candidate_indices=candidates)
        return train_score, joint_score
    return model.score(train_emissions, bin_centers), model.score(joint_emissions, bin_centers)


def _candidate_indices_for_model(model, emissions, bin_centers):
    try:
        return model.candidate_indices(emissions, bin_centers)
    except TypeError:
        return model.candidate_indices(emissions)


def _is_clusterless_model(model: object) -> bool:
    return isinstance(model, ClusterlessStateSpaceReplayModel)


def _clusterless_mark_likelihood(config: BenchmarkConfig) -> str:
    value = getattr(config, "clusterless_mark_likelihood", "local-kde")
    return _normalize_mark_likelihood(value)


def _clusterless_mark_config(config: BenchmarkConfig) -> ClusterlessMarkConfig:
    encoding_config = config.encoding
    return ClusterlessMarkConfig(
        encoding=encoding_config,
        mark_smoothing_sigma_bins=float(
            getattr(config, "clusterless_mark_smoothing_sigma_bins", 1.0)
        ),
        mark_prior_count=float(getattr(config, "clusterless_mark_prior_count", 1.0)),
        mark_variance_floor=float(getattr(config, "clusterless_mark_variance_floor", 1.0)),
        rate_floor_hz=float(getattr(config, "clusterless_rate_floor_hz", 1e-4)),
        use_excitatory=bool(encoding_config.use_excitatory),
        mark_likelihood=_clusterless_mark_likelihood(config),
        mark_kde_bandwidth=_optional_float(
            getattr(config, "clusterless_mark_kde_bandwidth", None)
        ),
        mark_kde_spatial_sigma_bins=_optional_float(
            getattr(config, "clusterless_mark_kde_spatial_sigma_bins", None)
        ),
        mark_kde_max_neighbors=int(
            getattr(config, "clusterless_mark_kde_max_neighbors", 256)
        ),
        mark_group_by=str(getattr(config, "clusterless_mark_group_by", "auto")),
    )


def _session_with_mark_cell_subset(
    session: ReplaySession,
    cell_ids: np.ndarray,
    *,
    role: str,
) -> ReplaySession:
    marks = session.spike_marks
    if marks is None or marks.n_features == 0:
        raise ValueError(
            "Clusterless held-out benchmarking requires spike marks; "
            f"session {session.session_id} has none."
        )
    if marks.cell_ids is None:
        raise ValueError(
            "Clusterless held-out benchmarking requires spike-mark cell IDs so "
            "train/test cell splits can be applied to the marked point process."
        )
    selected = np.asarray(cell_ids, dtype=int)
    if selected.size == 0:
        raise ValueError(f"No {role} cell IDs were selected for clusterless scoring.")
    mark_cell_ids = np.asarray(marks.cell_ids, dtype=int)
    keep = np.isin(mark_cell_ids, selected)
    if not np.any(keep):
        selected_text = _format_cell_ids(selected)
        raise ValueError(f"No {role} spike marks found for selected cell IDs: {selected_text}")
    filtered_marks = SpikeMarkData(
        times=np.asarray(marks.times, dtype=float)[keep].copy(),
        marks=np.asarray(marks.marks, dtype=float)[keep].copy(),
        source_file=marks.source_file,
        source_variable=marks.source_variable,
        feature_names=marks.feature_names,
        cell_ids=mark_cell_ids[keep].copy(),
        group_ids=None if marks.group_ids is None else np.asarray(marks.group_ids, dtype=int)[keep].copy(),
    )
    return replace(session, spike_marks=filtered_marks)


def _format_cell_ids(cell_ids: np.ndarray) -> str:
    return ",".join(str(int(cell_id)) for cell_id in np.asarray(cell_ids, dtype=int))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _benchmark_config_metadata(config: BenchmarkConfig) -> dict[str, object]:
    return {
        "benchmark_test_cell_fraction": float(config.test_cell_fraction),
        "benchmark_random_seed": int(config.random_seed),
        "encoding_bin_size_cm": float(config.encoding.bin_size_cm),
        "encoding_smoothing_sigma_bins": float(config.encoding.smoothing_sigma_bins),
        "encoding_min_speed_cm_s": float(config.encoding.min_speed_cm_s),
        "encoding_min_occupancy_s": float(config.encoding.min_occupancy_s),
        "encoding_rate_floor_hz": float(config.encoding.rate_floor_hz),
        "encoding_arena_padding_cm": float(config.encoding.arena_padding_cm),
        "encoding_use_excitatory": bool(config.encoding.use_excitatory),
        "emission_time_bin_s": float(config.emissions.time_bin_s),
        "emission_spike_rate_scale": float(config.emissions.spike_rate_scale),
        "emission_likelihood_temperature": float(config.emissions.likelihood_temperature),
        "emission_negative_binomial_overdispersion": float(config.emissions.negative_binomial_overdispersion),
        "clusterless_mark_smoothing_sigma_bins": float(
            getattr(config, "clusterless_mark_smoothing_sigma_bins", 1.0)
        ),
        "clusterless_mark_prior_count": float(
            getattr(config, "clusterless_mark_prior_count", 1.0)
        ),
        "clusterless_mark_variance_floor": float(
            getattr(config, "clusterless_mark_variance_floor", 1.0)
        ),
        "clusterless_rate_floor_hz": float(getattr(config, "clusterless_rate_floor_hz", 1e-4)),
        "clusterless_mark_likelihood": _clusterless_mark_likelihood(config),
        "clusterless_mark_kde_bandwidth": _optional_float(
            getattr(config, "clusterless_mark_kde_bandwidth", None)
        ),
        "clusterless_mark_kde_spatial_sigma_bins": _optional_float(
            getattr(config, "clusterless_mark_kde_spatial_sigma_bins", None)
        ),
        "clusterless_mark_kde_max_neighbors": int(
            getattr(config, "clusterless_mark_kde_max_neighbors", 256)
        ),
        "clusterless_mark_group_by": str(
            getattr(config, "clusterless_mark_group_by", "auto")
        ),
        "state_space_valid_occupancy_threshold_s": float(
            config.state_space_valid_occupancy_threshold_s
        ),
    }


def _session_mark_diagnostics(session: ReplaySession) -> dict[str, object]:
    marks = session.spike_marks
    return {
        "spike_mark_features": 0 if marks is None else marks.n_features,
        "spike_mark_source": "" if marks is None else f"{marks.source_file}:{marks.source_variable}",
        "clusterless_mark_likelihood_available": bool(marks is not None and marks.n_features > 0),
        "clusterless_tetrode_grouping_available": bool(marks is not None and marks.group_ids is not None),
        "clusterless_mark_groups": 0 if marks is None or marks.group_ids is None else int(np.unique(marks.group_ids).shape[0]),
    }


def _event_indices(session: ReplaySession, config: BenchmarkConfig) -> np.ndarray:
    if config.event_epoch == "run":
        indices = session.ripple_indices_in_run()
    elif config.event_epoch == "all":
        indices = np.arange(session.ripple_count, dtype=int)
    else:
        raise ValueError("event_epoch must be 'run' or 'all'")
    if config.max_events_per_session is not None:
        indices = indices[: config.max_events_per_session]
    return indices


def _split_cells(cell_ids: np.ndarray, test_fraction: float, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_cell_fraction must be in (0, 1)")
    rng = np.random.default_rng(random_seed)
    shuffled = np.asarray(cell_ids, dtype=int).copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(shuffled.size * test_fraction)))
    n_test = min(n_test, shuffled.size - 1) if shuffled.size > 1 else 0
    test = np.sort(shuffled[:n_test])
    train = np.sort(shuffled[n_test:])
    return train, test


def _build_models(
    config: BenchmarkConfig,
    session: ReplaySession | None = None,
) -> dict[str, object]:
    goal_candidates = _session_goal_candidates(session) if session is not None else None
    clusterless_kwargs = {
        "mark_likelihood": _clusterless_mark_likelihood(config),
    }
    available = {
        "random": RandomModel(),
        "stationary": StationaryModel(),
        "diffusion": CandidateKinematicModel(mode="diffusion", top_k=config.candidate_top_k),
        "momentum": CandidateKinematicModel(mode="momentum", top_k=config.candidate_top_k),
        "imm": CandidateKinematicModel(mode="imm", top_k=config.candidate_top_k),
        "sorted-spike-state-space-stationary": SortedSpikeStateSpaceReplayModel(mode="stationary"),
        "sorted-spike-state-space-diffusion": SortedSpikeStateSpaceReplayModel(mode="diffusion"),
        "sorted-spike-state-space-fragmented": SortedSpikeStateSpaceReplayModel(mode="fragmented"),
        "sorted-spike-state-space-jump": SortedSpikeStateSpaceReplayModel(mode="jump"),
        "sorted-spike-state-space-momentum": SortedSpikeStateSpaceReplayModel(mode="momentum"),
        "sorted-spike-state-space-imm": SortedSpikeStateSpaceReplayModel(mode="imm"),
        "state-space-stationary": SortedSpikeStateSpaceReplayModel(mode="stationary", name="state-space-stationary"),
        "state-space-diffusion": SortedSpikeStateSpaceReplayModel(mode="diffusion", name="state-space-diffusion"),
        "state-space-fragmented": SortedSpikeStateSpaceReplayModel(mode="fragmented", name="state-space-fragmented"),
        "state-space-jump": SortedSpikeStateSpaceReplayModel(mode="jump", name="state-space-jump"),
        "state-space-momentum": SortedSpikeStateSpaceReplayModel(mode="momentum", name="state-space-momentum"),
        "state-space-imm": SortedSpikeStateSpaceReplayModel(mode="imm", name="state-space-imm"),
        "clusterless-state-space-stationary": ClusterlessStateSpaceReplayModel(mode="stationary", **clusterless_kwargs),
        "clusterless-state-space-diffusion": ClusterlessStateSpaceReplayModel(mode="diffusion", **clusterless_kwargs),
        "clusterless-state-space-fragmented": ClusterlessStateSpaceReplayModel(mode="fragmented", **clusterless_kwargs),
        "clusterless-state-space-jump": ClusterlessStateSpaceReplayModel(mode="jump", **clusterless_kwargs),
        "clusterless-state-space-momentum": ClusterlessStateSpaceReplayModel(mode="momentum", **clusterless_kwargs),
        "clusterless-state-space-imm": ClusterlessStateSpaceReplayModel(mode="imm", **clusterless_kwargs),
        "pyrecest-goal-particle": PyRecEstGoalParticleModel(
            candidate_goals=goal_candidates,
            n_particles=config.pyrecest_particles,
            alpha=config.pyrecest_alpha,
            beta=config.pyrecest_beta,
            process_noise_sigma_cm_s=config.pyrecest_process_noise_sigma_cm_s,
            position_jump_sigma_cm=config.pyrecest_position_jump_sigma_cm,
            jump_probability=config.pyrecest_jump_probability,
            goal_reset_probability=config.pyrecest_goal_reset_probability,
            position_proposal_probability=config.pyrecest_position_proposal_probability,
            initial_velocity_sigma_cm_s=config.pyrecest_initial_velocity_sigma_cm_s,
            random_seed=config.random_seed,
        ),
        "pyrecest-goal-particle-imm": PyRecEstGoalParticleIMMModel(
            candidate_goals=goal_candidates,
            n_particles=config.pyrecest_particles,
            alpha=config.pyrecest_alpha,
            beta=config.pyrecest_beta,
            process_noise_sigma_cm_s=config.pyrecest_process_noise_sigma_cm_s,
            position_jump_sigma_cm=config.pyrecest_position_jump_sigma_cm,
            jump_probability=config.pyrecest_jump_probability,
            goal_reset_probability=config.pyrecest_goal_reset_probability,
            position_proposal_probability=config.pyrecest_position_proposal_probability,
            initial_velocity_sigma_cm_s=config.pyrecest_initial_velocity_sigma_cm_s,
            mode_stickiness=config.pyrecest_imm_mode_stickiness,
            stationary_velocity_decay=config.pyrecest_imm_stationary_velocity_decay,
            diffusion_velocity_decay=config.pyrecest_imm_diffusion_velocity_decay,
            momentum_velocity_decay=config.pyrecest_imm_momentum_velocity_decay,
            jump_fraction=config.pyrecest_imm_jump_fraction,
            jump_velocity_decay=config.pyrecest_imm_jump_velocity_decay,
            random_seed=config.random_seed,
        ),
    }
    if config.state_space_valid_occupancy_threshold_s > 0.0:
        for model in available.values():
            if isinstance(model, StateSpaceReplayModel):
                current = model.config or StateSpaceDecoderConfig(mode=model.mode)
                model.config = replace(
                    current,
                    valid_occupancy_threshold_s=config.state_space_valid_occupancy_threshold_s,
                )
    return {name: available[name] for name in config.models}


def _session_goal_candidates(session: ReplaySession | None) -> np.ndarray | None:
    if session is None:
        return None
    from .ground_truth import infer_well_locations

    wells = infer_well_locations(session)
    if wells.empty:
        return None
    return wells[["well_x", "well_y"]].to_numpy(dtype=float)


_BEST_STATIC_BASELINE_MODELS = frozenset(
    {
        "random",
        "stationary",
        "diffusion",
        "momentum",
    }
)

_STATE_SPACE_STATIC_BASELINE_MODES = frozenset(
    {
        "stationary",
        "diffusion",
        "fragmented",
        "jump",
        "momentum",
    }
)

_STATE_SPACE_STATIC_BASELINE_PREFIXES = (
    "sorted-spike-state-space-",
    "state-space-",
    "clusterless-state-space-",
)


def _is_best_static_baseline_model(model_name: object) -> bool:
    """Return whether a model belongs in the best-static held-out baseline."""

    model = str(model_name)
    if model in _BEST_STATIC_BASELINE_MODELS:
        return True
    for prefix in _STATE_SPACE_STATIC_BASELINE_PREFIXES:
        if model.startswith(prefix):
            mode = model.removeprefix(prefix)
            return mode in _STATE_SPACE_STATIC_BASELINE_MODES
    return False


def _add_relative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_evidence_support_columns(frame)
    static_mask = frame["model"].map(_is_best_static_baseline_model)
    exact_static_mask = static_mask & frame["evidence_comparable"].fillna(False).astype(bool)
    best_static = (
        frame[exact_static_mask]
        .groupby(["session", "event_index"])["heldout_log_likelihood"]
        .max()
        .rename("best_static_heldout_log_likelihood")
        .reset_index()
    )
    truncated_static_mask = static_mask & frame["evidence_support"].eq(TRUNCATED_EVIDENCE_SUPPORT)
    best_static_truncated_lower_bound = (
        frame[truncated_static_mask]
        .groupby(["session", "event_index"])["heldout_log_likelihood"]
        .max()
        .rename("best_static_truncated_lower_bound_heldout_log_likelihood")
        .reset_index()
    )
    merged = frame.merge(best_static, on=["session", "event_index"], how="left")
    merged = merged.merge(
        best_static_truncated_lower_bound,
        on=["session", "event_index"],
        how="left",
    )
    denom = pd.Series(
        np.maximum(merged["test_spikes"].to_numpy(dtype=float), 1.0),
        index=merged.index,
    )

    has_exact_static_baseline = merged["best_static_heldout_log_likelihood"].notna()
    exact_rows = merged["evidence_comparable"].fillna(False).astype(bool) & has_exact_static_baseline
    merged["delta_vs_best_static"] = np.nan
    merged.loc[exact_rows, "delta_vs_best_static"] = (
        merged.loc[exact_rows, "heldout_log_likelihood"]
        - merged.loc[exact_rows, "best_static_heldout_log_likelihood"]
    )
    merged["bits_per_spike_vs_best_static"] = np.nan
    merged.loc[exact_rows, "bits_per_spike_vs_best_static"] = (
        merged.loc[exact_rows, "delta_vs_best_static"] / np.log(2.0) / denom.loc[exact_rows]
    )

    truncated_rows = merged["evidence_support"].eq(TRUNCATED_EVIDENCE_SUPPORT)
    truncated_vs_exact_rows = truncated_rows & has_exact_static_baseline
    merged["lower_bound_delta_vs_best_static"] = np.nan
    merged.loc[truncated_vs_exact_rows, "lower_bound_delta_vs_best_static"] = (
        merged.loc[truncated_vs_exact_rows, "heldout_log_likelihood"]
        - merged.loc[truncated_vs_exact_rows, "best_static_heldout_log_likelihood"]
    )
    merged["lower_bound_bits_per_spike_vs_best_static"] = np.nan
    merged.loc[truncated_vs_exact_rows, "lower_bound_bits_per_spike_vs_best_static"] = (
        merged.loc[truncated_vs_exact_rows, "lower_bound_delta_vs_best_static"]
        / np.log(2.0)
        / denom.loc[truncated_vs_exact_rows]
    )

    truncated_vs_truncated_rows = truncated_rows & merged[
        "best_static_truncated_lower_bound_heldout_log_likelihood"
    ].notna()
    merged["delta_vs_best_static_truncated_lower_bound"] = np.nan
    merged.loc[truncated_vs_truncated_rows, "delta_vs_best_static_truncated_lower_bound"] = (
        merged.loc[truncated_vs_truncated_rows, "heldout_log_likelihood"]
        - merged.loc[
            truncated_vs_truncated_rows,
            "best_static_truncated_lower_bound_heldout_log_likelihood",
        ]
    )
    merged["bits_per_spike_vs_best_static_truncated_lower_bound"] = np.nan
    merged.loc[truncated_vs_truncated_rows, "bits_per_spike_vs_best_static_truncated_lower_bound"] = (
        merged.loc[truncated_vs_truncated_rows, "delta_vs_best_static_truncated_lower_bound"]
        / np.log(2.0)
        / denom.loc[truncated_vs_truncated_rows]
    )
    return merged
