"""Score-table metadata helpers shared by model-evidence and ground-truth code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .encoding import EmissionConfig, EncodingConfig


def encoding_config_for_scores(
    scores_frame: pd.DataFrame,
    fallback: EncodingConfig,
) -> EncodingConfig:
    """Build an encoding config from canonical or legacy score metadata.

    Held-out benchmark outputs use canonical ``encoding_*`` columns, whereas
    older model-evidence outputs used shorter names such as ``bin_size_cm``.
    Accept both, but reject conflicting values when both are present.
    """

    return EncodingConfig(
        bin_size_cm=_unique_float_from_columns(
            scores_frame,
            ("encoding_bin_size_cm", "bin_size_cm"),
            fallback.bin_size_cm,
        ),
        smoothing_sigma_bins=_unique_float_from_columns(
            scores_frame,
            ("encoding_smoothing_sigma_bins", "smoothing_sigma_bins"),
            fallback.smoothing_sigma_bins,
        ),
        min_speed_cm_s=_unique_float_from_columns(
            scores_frame,
            ("encoding_min_speed_cm_s", "min_speed_cm_s"),
            fallback.min_speed_cm_s,
        ),
        min_occupancy_s=_unique_float_from_columns(
            scores_frame,
            ("encoding_min_occupancy_s",),
            fallback.min_occupancy_s,
        ),
        rate_floor_hz=_unique_float_from_columns(
            scores_frame,
            ("encoding_rate_floor_hz",),
            fallback.rate_floor_hz,
        ),
        arena_padding_cm=_unique_float_from_columns(
            scores_frame,
            ("encoding_arena_padding_cm",),
            fallback.arena_padding_cm,
        ),
        use_excitatory=_unique_bool_from_column(
            scores_frame,
            "encoding_use_excitatory",
            fallback.use_excitatory,
        ),
    )


def emission_config_for_scores(
    scores_frame: pd.DataFrame,
    fallback: EmissionConfig,
) -> EmissionConfig:
    """Build an emission config from canonical or legacy score metadata."""

    return EmissionConfig(
        time_bin_s=_unique_float_from_columns(
            scores_frame,
            ("emission_time_bin_s", "time_bin_s"),
            fallback.time_bin_s,
        ),
        spike_rate_scale=_unique_float_from_columns(
            scores_frame,
            ("emission_spike_rate_scale", "spike_rate_scale"),
            fallback.spike_rate_scale,
        ),
        likelihood_temperature=_unique_float_from_columns(
            scores_frame,
            ("emission_likelihood_temperature", "likelihood_temperature"),
            fallback.likelihood_temperature,
        ),
        negative_binomial_overdispersion=_unique_float_from_columns(
            scores_frame,
            ("emission_negative_binomial_overdispersion", "negative_binomial_overdispersion"),
            fallback.negative_binomial_overdispersion,
        ),
    )


def apply_model_hyperparam_patch() -> None:
    """Make post-hoc decoding preserve model hyperparameters from score tables.

    The model-evidence script historically wrote encoding/emission metadata but
    not the candidate-model dynamics settings. Rather than requiring posterior
    artifacts, this patch makes candidate-model scores emit those settings as
    diagnostics and teaches ground-truth comparison to rebuild the models from
    canonical metadata, legacy columns, or diagnostic fallbacks.
    """

    from . import benchmarks as bench
    from . import ground_truth as gt
    from . import models as model_mod
    from .pyrecest_models import PyRecEstGoalParticleIMMModel, PyRecEstGoalParticleModel
    from .sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
    from .state_space import StateSpaceDecoderConfig

    if getattr(bench, "_model_hyperparam_patch_applied", False):
        return

    @dataclass(frozen=True)
    class BenchmarkConfig:
        encoding: EncodingConfig = field(default_factory=EncodingConfig)
        emissions: EmissionConfig = field(default_factory=EmissionConfig)
        test_cell_fraction: float = 0.25
        max_events_per_session: int | None = None
        candidate_top_k: int = 64
        stationary_sigma_cm: float = 2.0
        diffusion_sigma_cm: float = 12.0
        momentum_sigma_cm: float = 12.0
        velocity_decay: float = 0.95
        mode_stickiness: float = 0.94
        state_space_stationary_sigma_cm: float = 2.0
        state_space_diffusion_sigma_cm_sqrt_s: float = 85.0
        state_space_max_step_sigma: float = 4.0
        state_space_imm_mode_stickiness: float = 0.95
        state_space_momentum_sigma_cm_sqrt_s: float = 85.0
        state_space_momentum_initial_sigma_cm_sqrt_s: float = 85.0
        state_space_momentum_velocity_decay: float = 0.95
        state_space_momentum_candidate_top_k: int = 128
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

    def cfg(config, name: str, default):
        return getattr(config, name, default)

    def candidate_model(config, mode: str, name: str | None = None):
        return model_mod.CandidateKinematicModel(
            mode=mode,
            top_k=cfg(config, "candidate_top_k", 64),
            stationary_sigma_cm=cfg(config, "stationary_sigma_cm", 2.0),
            diffusion_sigma_cm=cfg(config, "diffusion_sigma_cm", 12.0),
            momentum_sigma_cm=cfg(config, "momentum_sigma_cm", 12.0),
            velocity_decay=cfg(config, "velocity_decay", 0.95),
            mode_stickiness=cfg(config, "mode_stickiness", 0.94),
            name=name,
        )

    def state_space_model(config, mode: str, name: str | None = None):
        return SortedSpikeStateSpaceReplayModel(
            mode=mode,
            name=name,
            config=StateSpaceDecoderConfig(
                mode=mode,
                stationary_sigma_cm=cfg(config, "state_space_stationary_sigma_cm", 2.0),
                diffusion_sigma_cm_sqrt_s=cfg(config, "state_space_diffusion_sigma_cm_sqrt_s", 85.0),
                max_step_sigma=cfg(config, "state_space_max_step_sigma", 4.0),
                imm_mode_stickiness=cfg(config, "state_space_imm_mode_stickiness", 0.95),
                momentum_sigma_cm_sqrt_s=cfg(config, "state_space_momentum_sigma_cm_sqrt_s", 85.0),
                momentum_initial_sigma_cm_sqrt_s=cfg(
                    config,
                    "state_space_momentum_initial_sigma_cm_sqrt_s",
                    85.0,
                ),
                momentum_velocity_decay=cfg(config, "state_space_momentum_velocity_decay", 0.95),
                momentum_candidate_top_k=cfg(config, "state_space_momentum_candidate_top_k", 128),
            ),
        )

    def build_models(config, session=None):
        goal_candidates = bench._session_goal_candidates(session) if session is not None else None
        available = {
            "random": model_mod.RandomModel(),
            "stationary": model_mod.StationaryModel(),
            "stationary-gaussian": candidate_model(config, "stationary", "stationary-gaussian"),
            "diffusion": candidate_model(config, "diffusion"),
            "momentum": candidate_model(config, "momentum"),
            "imm": candidate_model(config, "imm"),
            "sorted-spike-state-space-stationary": state_space_model(config, "stationary"),
            "sorted-spike-state-space-diffusion": state_space_model(config, "diffusion"),
            "sorted-spike-state-space-fragmented": state_space_model(config, "fragmented"),
            "sorted-spike-state-space-jump": state_space_model(config, "jump"),
            "sorted-spike-state-space-momentum": state_space_model(config, "momentum"),
            "sorted-spike-state-space-imm": state_space_model(config, "imm"),
            "state-space-stationary": state_space_model(config, "stationary", "state-space-stationary"),
            "state-space-diffusion": state_space_model(config, "diffusion", "state-space-diffusion"),
            "state-space-fragmented": state_space_model(config, "fragmented", "state-space-fragmented"),
            "state-space-jump": state_space_model(config, "jump", "state-space-jump"),
            "state-space-momentum": state_space_model(config, "momentum", "state-space-momentum"),
            "state-space-imm": state_space_model(config, "imm", "state-space-imm"),
            "pyrecest-goal-particle": PyRecEstGoalParticleModel(
                candidate_goals=goal_candidates,
                n_particles=cfg(config, "pyrecest_particles", 512),
                alpha=cfg(config, "pyrecest_alpha", 0.80),
                beta=cfg(config, "pyrecest_beta", 1.00),
                process_noise_sigma_cm_s=cfg(config, "pyrecest_process_noise_sigma_cm_s", 60.0),
                position_jump_sigma_cm=cfg(config, "pyrecest_position_jump_sigma_cm", 25.0),
                jump_probability=cfg(config, "pyrecest_jump_probability", 0.03),
                goal_reset_probability=cfg(config, "pyrecest_goal_reset_probability", 0.02),
                position_proposal_probability=cfg(config, "pyrecest_position_proposal_probability", 0.0),
                initial_velocity_sigma_cm_s=cfg(config, "pyrecest_initial_velocity_sigma_cm_s", 120.0),
                random_seed=cfg(config, "random_seed", 1),
            ),
            "pyrecest-goal-particle-imm": PyRecEstGoalParticleIMMModel(
                candidate_goals=goal_candidates,
                n_particles=cfg(config, "pyrecest_particles", 512),
                alpha=cfg(config, "pyrecest_alpha", 0.80),
                beta=cfg(config, "pyrecest_beta", 1.00),
                process_noise_sigma_cm_s=cfg(config, "pyrecest_process_noise_sigma_cm_s", 60.0),
                position_jump_sigma_cm=cfg(config, "pyrecest_position_jump_sigma_cm", 25.0),
                jump_probability=cfg(config, "pyrecest_jump_probability", 0.03),
                goal_reset_probability=cfg(config, "pyrecest_goal_reset_probability", 0.02),
                position_proposal_probability=cfg(config, "pyrecest_position_proposal_probability", 0.0),
                initial_velocity_sigma_cm_s=cfg(config, "pyrecest_initial_velocity_sigma_cm_s", 120.0),
                mode_stickiness=cfg(config, "pyrecest_imm_mode_stickiness", 0.95),
                stationary_velocity_decay=cfg(config, "pyrecest_imm_stationary_velocity_decay", 0.0),
                diffusion_velocity_decay=cfg(config, "pyrecest_imm_diffusion_velocity_decay", 0.0),
                momentum_velocity_decay=cfg(config, "pyrecest_imm_momentum_velocity_decay", 0.95),
                jump_fraction=cfg(config, "pyrecest_imm_jump_fraction", 0.9),
                jump_velocity_decay=cfg(config, "pyrecest_imm_jump_velocity_decay", 0.25),
                random_seed=cfg(config, "random_seed", 1),
            ),
        }
        return {name: available[name] for name in config.models}

    def benchmark_config_metadata(config) -> dict[str, object]:
        base = {
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
            "candidate_top_k": int(cfg(config, "candidate_top_k", 64)),
            "candidate_stationary_sigma_cm": float(cfg(config, "stationary_sigma_cm", 2.0)),
            "candidate_diffusion_sigma_cm": float(cfg(config, "diffusion_sigma_cm", 12.0)),
            "candidate_momentum_sigma_cm": float(cfg(config, "momentum_sigma_cm", 12.0)),
            "candidate_velocity_decay": float(cfg(config, "velocity_decay", 0.95)),
            "candidate_mode_stickiness": float(cfg(config, "mode_stickiness", 0.94)),
            "state_space_stationary_sigma_cm": float(cfg(config, "state_space_stationary_sigma_cm", 2.0)),
            "state_space_diffusion_sigma_cm_sqrt_s": float(cfg(config, "state_space_diffusion_sigma_cm_sqrt_s", 85.0)),
            "state_space_max_step_sigma": float(cfg(config, "state_space_max_step_sigma", 4.0)),
            "state_space_imm_mode_stickiness": float(cfg(config, "state_space_imm_mode_stickiness", 0.95)),
            "state_space_momentum_sigma_cm_sqrt_s": float(cfg(config, "state_space_momentum_sigma_cm_sqrt_s", 85.0)),
            "state_space_momentum_initial_sigma_cm_sqrt_s": float(cfg(config, "state_space_momentum_initial_sigma_cm_sqrt_s", 85.0)),
            "state_space_momentum_velocity_decay": float(cfg(config, "state_space_momentum_velocity_decay", 0.95)),
            "state_space_momentum_candidate_top_k": int(cfg(config, "state_space_momentum_candidate_top_k", 128)),
        }
        return base

    original_candidate_score = model_mod.CandidateKinematicModel.score
    if not getattr(original_candidate_score, "_hyperparam_metadata_wrapped", False):

        def candidate_score_with_metadata(self, emissions, bin_centers, candidate_indices=None):
            result = original_candidate_score(self, emissions, bin_centers, candidate_indices=candidate_indices)
            result.diagnostics.update(
                {
                    "candidate_top_k": int(self.top_k),
                    "candidate_stationary_sigma_cm": float(self.stationary_sigma_cm),
                    "candidate_diffusion_sigma_cm": float(self.diffusion_sigma_cm),
                    "candidate_momentum_sigma_cm": float(self.momentum_sigma_cm),
                    "candidate_velocity_decay": float(self.velocity_decay),
                    "candidate_mode_stickiness": float(self.mode_stickiness),
                }
            )
            return result

        candidate_score_with_metadata._hyperparam_metadata_wrapped = True
        model_mod.CandidateKinematicModel.score = candidate_score_with_metadata

    def benchmark_config_for_scores(
        scores_frame: pd.DataFrame,
        *,
        encoding_config: EncodingConfig,
        emission_config: EmissionConfig,
        test_cell_fraction: float,
        candidate_top_k: int,
        stationary_sigma_cm: float,
        diffusion_sigma_cm: float,
        momentum_sigma_cm: float,
        velocity_decay: float,
        mode_stickiness: float,
        state_space_stationary_sigma_cm: float,
        state_space_diffusion_sigma_cm_sqrt_s: float,
        state_space_max_step_sigma: float,
        state_space_imm_mode_stickiness: float,
        state_space_momentum_sigma_cm_sqrt_s: float,
        state_space_momentum_initial_sigma_cm_sqrt_s: float,
        state_space_momentum_velocity_decay: float,
        state_space_momentum_candidate_top_k: int,
        pyrecest_particles: int,
        pyrecest_alpha: float,
        pyrecest_beta: float,
        pyrecest_process_noise_sigma_cm_s: float,
        pyrecest_position_jump_sigma_cm: float,
        pyrecest_jump_probability: float,
        pyrecest_goal_reset_probability: float,
        pyrecest_position_proposal_probability: float,
        pyrecest_initial_velocity_sigma_cm_s: float,
        pyrecest_imm_mode_stickiness: float,
        pyrecest_imm_stationary_velocity_decay: float,
        pyrecest_imm_diffusion_velocity_decay: float,
        pyrecest_imm_momentum_velocity_decay: float,
        pyrecest_imm_jump_fraction: float,
        pyrecest_imm_jump_velocity_decay: float,
        random_seed: int,
        model_names: tuple[str, ...],
    ) -> BenchmarkConfig:
        return BenchmarkConfig(
            encoding=encoding_config,
            emissions=emission_config,
            test_cell_fraction=_unique_float_from_column(scores_frame, "benchmark_test_cell_fraction", test_cell_fraction),
            candidate_top_k=_unique_int_from_columns(scores_frame, ("candidate_top_k", "diagnostic_candidate_top_k"), candidate_top_k),
            stationary_sigma_cm=_unique_float_from_columns(scores_frame, ("candidate_stationary_sigma_cm", "stationary_sigma_cm", "diagnostic_candidate_stationary_sigma_cm"), stationary_sigma_cm),
            diffusion_sigma_cm=_unique_float_from_columns(scores_frame, ("candidate_diffusion_sigma_cm", "diffusion_sigma_cm", "diagnostic_candidate_diffusion_sigma_cm"), diffusion_sigma_cm),
            momentum_sigma_cm=_unique_float_from_columns(scores_frame, ("candidate_momentum_sigma_cm", "momentum_sigma_cm", "diagnostic_candidate_momentum_sigma_cm"), momentum_sigma_cm),
            velocity_decay=_unique_float_from_columns(scores_frame, ("candidate_velocity_decay", "velocity_decay", "diagnostic_candidate_velocity_decay"), velocity_decay),
            mode_stickiness=_unique_float_from_columns(scores_frame, ("candidate_mode_stickiness", "mode_stickiness", "diagnostic_candidate_mode_stickiness"), mode_stickiness),
            state_space_stationary_sigma_cm=_unique_float_from_columns(scores_frame, ("state_space_stationary_sigma_cm", "diagnostic_state_space_stationary_sigma_cm"), state_space_stationary_sigma_cm),
            state_space_diffusion_sigma_cm_sqrt_s=_unique_float_from_columns(scores_frame, ("state_space_diffusion_sigma_cm_sqrt_s", "diagnostic_state_space_diffusion_sigma_cm_sqrt_s"), state_space_diffusion_sigma_cm_sqrt_s),
            state_space_max_step_sigma=_unique_float_from_columns(scores_frame, ("state_space_max_step_sigma", "diagnostic_state_space_max_step_sigma"), state_space_max_step_sigma),
            state_space_imm_mode_stickiness=_unique_float_from_columns(scores_frame, ("state_space_imm_mode_stickiness", "diagnostic_state_space_imm_mode_stickiness"), state_space_imm_mode_stickiness),
            state_space_momentum_sigma_cm_sqrt_s=_unique_float_from_columns(scores_frame, ("state_space_momentum_sigma_cm_sqrt_s", "diagnostic_state_space_momentum_sigma_cm_sqrt_s"), state_space_momentum_sigma_cm_sqrt_s),
            state_space_momentum_initial_sigma_cm_sqrt_s=_unique_float_from_columns(scores_frame, ("state_space_momentum_initial_sigma_cm_sqrt_s", "diagnostic_state_space_momentum_initial_sigma_cm_sqrt_s"), state_space_momentum_initial_sigma_cm_sqrt_s),
            state_space_momentum_velocity_decay=_unique_float_from_columns(scores_frame, ("state_space_momentum_velocity_decay", "diagnostic_state_space_momentum_velocity_decay"), state_space_momentum_velocity_decay),
            state_space_momentum_candidate_top_k=_unique_int_from_columns(scores_frame, ("state_space_momentum_candidate_top_k", "diagnostic_state_space_momentum_candidate_top_k", "diagnostic_state_space_imm_candidate_top_k"), state_space_momentum_candidate_top_k),
            pyrecest_particles=pyrecest_particles,
            pyrecest_alpha=pyrecest_alpha,
            pyrecest_beta=pyrecest_beta,
            pyrecest_process_noise_sigma_cm_s=pyrecest_process_noise_sigma_cm_s,
            pyrecest_position_jump_sigma_cm=pyrecest_position_jump_sigma_cm,
            pyrecest_jump_probability=pyrecest_jump_probability,
            pyrecest_goal_reset_probability=pyrecest_goal_reset_probability,
            pyrecest_position_proposal_probability=pyrecest_position_proposal_probability,
            pyrecest_initial_velocity_sigma_cm_s=pyrecest_initial_velocity_sigma_cm_s,
            pyrecest_imm_mode_stickiness=pyrecest_imm_mode_stickiness,
            pyrecest_imm_stationary_velocity_decay=pyrecest_imm_stationary_velocity_decay,
            pyrecest_imm_diffusion_velocity_decay=pyrecest_imm_diffusion_velocity_decay,
            pyrecest_imm_momentum_velocity_decay=pyrecest_imm_momentum_velocity_decay,
            pyrecest_imm_jump_fraction=pyrecest_imm_jump_fraction,
            pyrecest_imm_jump_velocity_decay=pyrecest_imm_jump_velocity_decay,
            random_seed=_unique_int_from_column(scores_frame, "benchmark_random_seed", random_seed),
            models=model_names,
        )

    def compare_scores_to_ground_truth(
        root: str | Path,
        scores: str | Path | pd.DataFrame,
        *,
        ground_truth: str | Path | pd.DataFrame | None = None,
        ground_truth_config=None,
        encoding_config: EncodingConfig | None = None,
        emission_config: EmissionConfig | None = None,
        test_cell_fraction: float = 0.25,
        candidate_top_k: int = 64,
        stationary_sigma_cm: float = 2.0,
        diffusion_sigma_cm: float = 12.0,
        momentum_sigma_cm: float = 12.0,
        velocity_decay: float = 0.95,
        mode_stickiness: float = 0.94,
        state_space_stationary_sigma_cm: float = 2.0,
        state_space_diffusion_sigma_cm_sqrt_s: float = 85.0,
        state_space_max_step_sigma: float = 4.0,
        state_space_imm_mode_stickiness: float = 0.95,
        state_space_momentum_sigma_cm_sqrt_s: float = 85.0,
        state_space_momentum_initial_sigma_cm_sqrt_s: float = 85.0,
        state_space_momentum_velocity_decay: float = 0.95,
        state_space_momentum_candidate_top_k: int = 128,
        pyrecest_particles: int = 512,
        pyrecest_alpha: float = 0.80,
        pyrecest_beta: float = 1.00,
        pyrecest_process_noise_sigma_cm_s: float = 60.0,
        pyrecest_position_jump_sigma_cm: float = 25.0,
        pyrecest_jump_probability: float = 0.03,
        pyrecest_goal_reset_probability: float = 0.02,
        pyrecest_position_proposal_probability: float = 0.0,
        pyrecest_initial_velocity_sigma_cm_s: float = 120.0,
        pyrecest_imm_mode_stickiness: float = 0.95,
        pyrecest_imm_stationary_velocity_decay: float = 0.0,
        pyrecest_imm_diffusion_velocity_decay: float = 0.0,
        pyrecest_imm_momentum_velocity_decay: float = 0.95,
        pyrecest_imm_jump_fraction: float = 0.9,
        pyrecest_imm_jump_velocity_decay: float = 0.25,
        random_seed: int = 1,
    ) -> pd.DataFrame:
        scores_frame = pd.read_csv(scores) if not isinstance(scores, pd.DataFrame) else scores.copy()
        gt_frame = gt._load_or_generate_ground_truth(root, ground_truth, ground_truth_config)
        if scores_frame.empty:
            return scores_frame

        benchmark_decode = gt._score_table_is_heldout_benchmark(scores_frame)
        encoding_config = encoding_config_for_scores(
            scores_frame,
            EncodingConfig() if encoding_config is None else encoding_config,
        )
        emission_config = emission_config_for_scores(
            scores_frame,
            EmissionConfig() if emission_config is None else emission_config,
        )
        model_names = gt._model_names_for_scores(scores_frame)
        model_config = benchmark_config_for_scores(
            scores_frame,
            encoding_config=encoding_config,
            emission_config=emission_config,
            test_cell_fraction=test_cell_fraction,
            candidate_top_k=candidate_top_k,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigma_cm=diffusion_sigma_cm,
            momentum_sigma_cm=momentum_sigma_cm,
            velocity_decay=velocity_decay,
            mode_stickiness=mode_stickiness,
            state_space_stationary_sigma_cm=state_space_stationary_sigma_cm,
            state_space_diffusion_sigma_cm_sqrt_s=state_space_diffusion_sigma_cm_sqrt_s,
            state_space_max_step_sigma=state_space_max_step_sigma,
            state_space_imm_mode_stickiness=state_space_imm_mode_stickiness,
            state_space_momentum_sigma_cm_sqrt_s=state_space_momentum_sigma_cm_sqrt_s,
            state_space_momentum_initial_sigma_cm_sqrt_s=state_space_momentum_initial_sigma_cm_sqrt_s,
            state_space_momentum_velocity_decay=state_space_momentum_velocity_decay,
            state_space_momentum_candidate_top_k=state_space_momentum_candidate_top_k,
            pyrecest_particles=pyrecest_particles,
            pyrecest_alpha=pyrecest_alpha,
            pyrecest_beta=pyrecest_beta,
            pyrecest_process_noise_sigma_cm_s=pyrecest_process_noise_sigma_cm_s,
            pyrecest_position_jump_sigma_cm=pyrecest_position_jump_sigma_cm,
            pyrecest_jump_probability=pyrecest_jump_probability,
            pyrecest_goal_reset_probability=pyrecest_goal_reset_probability,
            pyrecest_position_proposal_probability=pyrecest_position_proposal_probability,
            pyrecest_initial_velocity_sigma_cm_s=pyrecest_initial_velocity_sigma_cm_s,
            pyrecest_imm_mode_stickiness=pyrecest_imm_mode_stickiness,
            pyrecest_imm_stationary_velocity_decay=pyrecest_imm_stationary_velocity_decay,
            pyrecest_imm_diffusion_velocity_decay=pyrecest_imm_diffusion_velocity_decay,
            pyrecest_imm_momentum_velocity_decay=pyrecest_imm_momentum_velocity_decay,
            pyrecest_imm_jump_fraction=pyrecest_imm_jump_fraction,
            pyrecest_imm_jump_velocity_decay=pyrecest_imm_jump_velocity_decay,
            random_seed=random_seed,
            model_names=model_names,
        )

        sessions = {session.session_id: session for session in gt.load_open_field_sessions(root)}
        decoded_rows: list[dict[str, object]] = []
        for session_id, session_scores in scores_frame.groupby("session", sort=False):
            session = sessions.get(str(session_id))
            if session is None:
                continue
            models = gt._build_models(model_config, session=session)
            wells = gt.infer_well_locations(session, ground_truth_config)
            encoding = gt.fit_place_field_encoding(session, encoding_config)
            if benchmark_decode:
                train_cells, test_cells = gt._cell_split_for_score_rows(session_scores, encoding, model_config)
                train_encoding = encoding.select_cells(train_cells)
                joint_encoding = encoding.select_cells(np.concatenate([train_cells, test_cells]))
            for event_index, event_scores in session_scores.groupby("event_index", sort=False):
                if benchmark_decode:
                    train_emissions = gt.build_emissions(session, train_encoding, int(event_index), emission_config)
                    joint_emissions = gt.build_emissions(session, joint_encoding, int(event_index), emission_config)
                    if train_emissions.n_time == 0 or joint_emissions.n_time == 0:
                        continue
                else:
                    emissions = gt.build_emissions(session, encoding, int(event_index), emission_config)
                    if emissions.n_time == 0:
                        continue
                for score_row in event_scores.itertuples(index=False):
                    model_name = str(getattr(score_row, "model"))
                    requested_model = gt._requested_model_name(score_row, model_name)
                    model = models.get(requested_model) or models.get(model_name)
                    if model is None:
                        continue
                    if benchmark_decode:
                        score = gt._score_joint_for_ground_truth(
                            model,
                            train_emissions,
                            joint_emissions,
                            encoding.bin_centers,
                        )
                    else:
                        score = model.score(emissions, encoding.bin_centers)
                    decoded_rows.append(
                        gt._decoded_row(
                            str(session_id),
                            int(event_index),
                            model_name,
                            score.terminal_log_posterior,
                            encoding.bin_centers,
                            wells,
                        )
                    )
        decoded = pd.DataFrame(decoded_rows)
        comparison = scores_frame.merge(gt_frame, on=["session", "event_index"], how="left")
        comparison = comparison.merge(decoded, on=["session", "event_index", "model"], how="left")
        return gt._add_ground_truth_metrics(comparison, decoded, gt_frame)

    bench.BenchmarkConfig = BenchmarkConfig
    gt.BenchmarkConfig = BenchmarkConfig
    bench._build_models = build_models
    gt._build_models = build_models
    bench._benchmark_config_metadata = benchmark_config_metadata
    gt.compare_scores_to_ground_truth = compare_scores_to_ground_truth
    bench._model_hyperparam_patch_applied = True


def _unique_float_from_column(frame: pd.DataFrame, column: str, default: float) -> float:
    return _unique_float_from_columns(frame, (column,), default)


def _unique_float_from_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    default: float,
) -> float:
    values: list[float] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            text = str(value).strip()
            if text:
                values.append(float(value))
    if not values:
        return float(default)
    first = values[0]
    if any(not np.isclose(value, first) for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return float(first)


def _unique_int_from_column(frame: pd.DataFrame, column: str, default: int) -> int:
    return _unique_int_from_columns(frame, (column,), default)


def _unique_int_from_columns(frame: pd.DataFrame, columns: tuple[str, ...], default: int) -> int:
    values: list[int] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            text = str(value).strip()
            if text:
                values.append(int(float(value)))
    if not values:
        return int(default)
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return int(first)


def _unique_bool_from_column(frame: pd.DataFrame, column: str, default: bool) -> bool:
    values: list[bool] = []
    if column in frame.columns:
        for value in frame[column].dropna():
            text = str(value).strip()
            if text:
                values.append(_parse_bool(value))
    if not values:
        return bool(default)
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{column} contains multiple values")
    return bool(first)


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)) and not np.isnan(value):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")
