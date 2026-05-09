"""Open-field held-out likelihood benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ReplaySession, load_open_field_sessions
from .encoding import EmissionConfig, EncodingConfig, build_emissions, fit_place_field_encoding
from .models import CandidateKinematicModel, RandomModel, StationaryModel
from .pyrecest_models import PyRecEstGoalParticleIMMModel, PyRecEstGoalParticleModel
from .sorted_spike_state_space import SortedSpikeStateSpaceReplayModel


@dataclass(frozen=True)
class BenchmarkConfig:
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    emissions: EmissionConfig = field(default_factory=EmissionConfig)
    test_cell_fraction: float = 0.25
    max_events_per_session: int | None = None
    candidate_top_k: int = 64
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
    random_seed: int = 1
    event_epoch: str = "run"
    models: tuple[str, ...] = ("random", "stationary", "diffusion", "momentum", "imm")


@dataclass
class BenchmarkResult:
    rows: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        if self.rows.empty:
            return pd.DataFrame()
        grouped = self.rows.groupby("model", as_index=False)
        return grouped.agg(
            events=("heldout_log_likelihood", "count"),
            mean_heldout_log_likelihood=("heldout_log_likelihood", "mean"),
            mean_delta_vs_best_static=("delta_vs_best_static", "mean"),
            mean_bits_per_spike_vs_best_static=("bits_per_spike_vs_best_static", "mean"),
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
    train_cells, test_cells = _split_cells(encoding.cell_ids, config.test_cell_fraction, config.random_seed)
    if test_cells.size == 0 or train_cells.size == 0:
        return []
    train_encoding = encoding.select_cells(train_cells)
    joint_encoding = encoding.select_cells(np.concatenate([train_cells, test_cells]))
    event_indices = _event_indices(session, config)
    model_objects = _build_models(config, session=session)
    rows: list[dict[str, object]] = []
    for event_index in event_indices:
        train_emissions = build_emissions(session, train_encoding, int(event_index), config.emissions)
        joint_emissions = build_emissions(session, joint_encoding, int(event_index), config.emissions)
        if train_emissions.n_time == 0 or joint_emissions.n_time == 0:
            continue
        for model_name, model in model_objects.items():
            if isinstance(model, CandidateKinematicModel):
                candidates = model.candidate_indices(train_emissions)
                train_score = model.score(train_emissions, encoding.bin_centers, candidate_indices=candidates)
                joint_score = model.score(joint_emissions, encoding.bin_centers, candidate_indices=candidates)
            else:
                train_score = model.score(train_emissions, encoding.bin_centers)
                joint_score = model.score(joint_emissions, encoding.bin_centers)
            heldout = joint_score.log_likelihood - train_score.log_likelihood
            rows.append(
                {
                    "session": session.session_id,
                    "event_index": int(event_index),
                    "model": model_name,
                    "heldout_log_likelihood": float(heldout),
                    "joint_log_likelihood": float(joint_score.log_likelihood),
                    "train_log_likelihood": float(train_score.log_likelihood),
                    "test_spikes": int(joint_emissions.n_spikes - train_emissions.n_spikes),
                    "n_time": int(train_emissions.n_time),
                    **{
                        f"diagnostic_{key}": value
                        for key, value in joint_score.diagnostics.items()
                    },
                }
            )
    return rows


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
    return {name: available[name] for name in config.models}


def _session_goal_candidates(session: ReplaySession | None) -> np.ndarray | None:
    if session is None:
        return None
    from .ground_truth import infer_well_locations

    wells = infer_well_locations(session)
    if wells.empty:
        return None
    return wells[["well_x", "well_y"]].to_numpy(dtype=float)


def _add_relative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    static_models = {"random", "stationary", "diffusion", "momentum"}
    best_static = (
        frame[frame["model"].isin(static_models)]
        .groupby(["session", "event_index"])["heldout_log_likelihood"]
        .max()
        .rename("best_static_heldout_log_likelihood")
        .reset_index()
    )
    merged = frame.merge(best_static, on=["session", "event_index"], how="left")
    merged["delta_vs_best_static"] = (
        merged["heldout_log_likelihood"] - merged["best_static_heldout_log_likelihood"]
    )
    denom = np.maximum(merged["test_spikes"].to_numpy(dtype=float), 1.0)
    merged["bits_per_spike_vs_best_static"] = merged["delta_vs_best_static"] / np.log(2.0) / denom
    return merged
