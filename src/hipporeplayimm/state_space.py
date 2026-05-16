"""State-space replay decoder baselines with full trajectory posteriors."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .encoding import LogEmissionTensor
from .models import EventScore, LOG_ZERO, _normalize_log_weights, _posterior_diagnostics
from .state_space_candidates import (
    _advance_imm_pair_log_alpha,
    _backward_imm_pair,
    _backward_imm_pair_for_mode,
    _init_imm_pair_log_alpha,
    _score_imm_candidates,
)
from .state_space_candidates_momentum import (
    _advance_momentum_pair,
    _backward_momentum_pair,
    _init_pair_log_alpha,
    _score_momentum_candidates,
)
from .state_space_first_order import (
    _apply_transition,
    _apply_transition_backward,
    _forward_backward_first_order,
    _score_first_order_imm,
    _score_fragmented,
    _score_stationary,
)
from .state_space_utils import (
    _as_log_probs,
    _candidate_log_masses,
    _full_grid_normalized_pairwise_gaussian_log_prob,
    _gaussian_transition_matrix,
    _mean_entropy,
    _mode_transition_matrix,
    _pairwise_gaussian_log_prob,
    _per_bin_sigma,
    _scaled_emissions,
    _top_candidate_indices,
    _validate_candidate_indices,
)


@dataclass(frozen=True)
class StateSpaceDecoderConfig:
    """Configuration for state-space replay baselines."""

    mode: str = "diffusion"
    stationary_sigma_cm: float = 2.0
    diffusion_sigma_cm_sqrt_s: float = 85.0
    max_step_sigma: float = 4.0
    imm_mode_stickiness: float = 0.95
    momentum_sigma_cm_sqrt_s: float = 85.0
    momentum_initial_sigma_cm_sqrt_s: float = 85.0
    momentum_velocity_decay: float = 0.95
    momentum_candidate_top_k: int = 128


@dataclass
class StateSpaceReplayModel:
    """Replay decoder baseline returning a posterior for every replay bin."""

    mode: str = "diffusion"
    config: StateSpaceDecoderConfig | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        allowed = {
            "stationary",
            "diffusion",
            "fragmented",
            "jump",
            "first-order-imm",
            "imm",
            "momentum",
        }
        if self.mode not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}")
        if self.name is None:
            self.name = f"state-space-{self.mode}"
        if self.config is None:
            self.config = StateSpaceDecoderConfig(mode=self.mode)
        elif self.config.mode != self.mode:
            self.config = replace(self.config, mode=self.mode)

    def candidate_indices(self, emissions: LogEmissionTensor) -> list[np.ndarray]:
        """Return the candidate support used by pruned momentum/IMM recursions."""

        assert self.config is not None
        return [_top_candidate_indices(row, self.config.momentum_candidate_top_k) for row in emissions.log_likelihood]

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray] | None = None,
    ) -> EventScore:
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        if emissions.n_bins != bin_centers.shape[0]:
            raise ValueError("emissions.n_bins must match bin_centers rows")
        assert self.config is not None

        if self.mode == "stationary":
            logp, trajectory = _score_stationary(emissions)
            transition_sigma_cm = 0.0
            extra: dict[str, float | int | str] = {}
        elif self.mode in {"fragmented", "jump"}:
            logp, trajectory = _score_fragmented(emissions)
            transition_sigma_cm = float("inf")
            extra = {}
        elif self.mode == "diffusion":
            transition_sigma_cm = _per_bin_sigma(self.config.diffusion_sigma_cm_sqrt_s, emissions.dt)
            transition = _gaussian_transition_matrix(bin_centers, transition_sigma_cm, self.config.max_step_sigma)
            logp, trajectory = _forward_backward_first_order(emissions.log_likelihood, transition)
            extra = {}
        elif self.mode == "first-order-imm":
            transition_sigma_cm = _per_bin_sigma(self.config.diffusion_sigma_cm_sqrt_s, emissions.dt)
            logp, trajectory, mode_post = _score_first_order_imm(
                emissions.log_likelihood,
                bin_centers,
                stationary_sigma_cm=self.config.stationary_sigma_cm,
                diffusion_sigma_cm=transition_sigma_cm,
                max_step_sigma=self.config.max_step_sigma,
                mode_stickiness=self.config.imm_mode_stickiness,
            )
            names = ("stationary", "diffusion", "fragmented")
            extra = {
                f"state_space_mode_{name}_terminal_probability": float(mode_post[-1, idx])
                for idx, name in enumerate(names)
            }
            extra.update(
                {
                    "state_space_imm_modes": ",".join(names),
                    "state_space_imm_evidence_support": "exact_full_grid",
                }
            )
        elif self.mode == "imm":
            transition_sigma_cm = _per_bin_sigma(self.config.diffusion_sigma_cm_sqrt_s, emissions.dt)
            momentum_transition_sigma_cm = _per_bin_sigma(self.config.momentum_sigma_cm_sqrt_s, emissions.dt)
            momentum_initial_sigma_cm = _per_bin_sigma(self.config.momentum_initial_sigma_cm_sqrt_s, emissions.dt)
            candidates = self.candidate_indices(emissions) if candidate_indices is None else candidate_indices
            _validate_candidate_indices(candidates, emissions.n_time, emissions.n_bins)
            logp, trajectory, mode_post, masses = _score_imm_candidates(
                emissions,
                bin_centers,
                stationary_sigma_cm=self.config.stationary_sigma_cm,
                diffusion_sigma_cm=transition_sigma_cm,
                momentum_sigma_cm=momentum_transition_sigma_cm,
                momentum_initial_sigma_cm=momentum_initial_sigma_cm,
                velocity_decay=self.config.momentum_velocity_decay,
                mode_stickiness=self.config.imm_mode_stickiness,
                candidate_indices=candidates,
            )
            names = ("stationary", "diffusion", "momentum", "jump")
            extra = {
                f"state_space_mode_{name}_terminal_probability": float(mode_post[-1, idx])
                for idx, name in enumerate(names)
            }
            extra.update(
                {
                    "mean_candidate_log_mass": float(np.mean(masses)),
                    "state_space_imm_modes": ",".join(names),
                    "state_space_imm_candidate_top_k": int(self.config.momentum_candidate_top_k),
                    "state_space_imm_candidate_support": "derived" if candidate_indices is None else "provided",
                    "state_space_imm_trajectory_posterior": "smoothed_pair_marginal",
                    "state_space_imm_evidence_support": "truncated_full_grid",
                    "state_space_momentum_transition_sigma_cm": float(momentum_transition_sigma_cm),
                    "state_space_momentum_initial_transition_sigma_cm": float(momentum_initial_sigma_cm),
                }
            )
        elif self.mode == "momentum":
            transition_sigma_cm = _per_bin_sigma(self.config.momentum_sigma_cm_sqrt_s, emissions.dt)
            candidates = self.candidate_indices(emissions) if candidate_indices is None else candidate_indices
            _validate_candidate_indices(candidates, emissions.n_time, emissions.n_bins)
            logp, trajectory, masses = _score_momentum_candidates(
                emissions,
                bin_centers,
                candidates,
                sigma_cm=transition_sigma_cm,
                initial_sigma_cm=_per_bin_sigma(self.config.momentum_initial_sigma_cm_sqrt_s, emissions.dt),
                velocity_decay=self.config.momentum_velocity_decay,
            )
            extra = {
                "mean_candidate_log_mass": float(np.mean(masses)),
                "state_space_momentum_candidate_top_k": int(self.config.momentum_candidate_top_k),
                "state_space_momentum_candidate_support": "derived" if candidate_indices is None else "provided",
                "state_space_momentum_trajectory_posterior": "smoothed_pair_marginal",
                "state_space_momentum_evidence_support": "truncated_full_grid",
            }
        else:  # pragma: no cover - __post_init__ validates this.
            raise ValueError(f"Unsupported state-space mode: {self.mode}")

        terminal = trajectory[-1]
        diagnostics: dict[str, float | int | str] = {
            "state_space_mode": str(self.mode),
            "state_space_time_bin_s": float(emissions.dt),
            "state_space_trajectory_posterior": 1,
            "state_space_trajectory_time_bins": int(emissions.n_time),
            "state_space_stationary_sigma_cm": float(self.config.stationary_sigma_cm),
            "state_space_diffusion_sigma_cm_sqrt_s": float(self.config.diffusion_sigma_cm_sqrt_s),
            "state_space_max_step_sigma": float(self.config.max_step_sigma),
            "state_space_imm_mode_stickiness": float(self.config.imm_mode_stickiness),
            "state_space_momentum_sigma_cm_sqrt_s": float(self.config.momentum_sigma_cm_sqrt_s),
            "state_space_momentum_initial_sigma_cm_sqrt_s": float(self.config.momentum_initial_sigma_cm_sqrt_s),
            "state_space_momentum_velocity_decay": float(self.config.momentum_velocity_decay),
            "state_space_transition_sigma_cm": float(transition_sigma_cm),
            "mean_trajectory_posterior_entropy": _mean_entropy(trajectory),
            **extra,
        }
        diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            str(self.name),
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )


__all__ = [
    "EventScore",
    "LOG_ZERO",
    "StateSpaceDecoderConfig",
    "StateSpaceReplayModel",
    "_advance_imm_pair_log_alpha",
    "_advance_momentum_pair",
    "_apply_transition",
    "_apply_transition_backward",
    "_as_log_probs",
    "_backward_imm_pair",
    "_backward_imm_pair_for_mode",
    "_backward_momentum_pair",
    "_candidate_log_masses",
    "_forward_backward_first_order",
    "_full_grid_normalized_pairwise_gaussian_log_prob",
    "_gaussian_transition_matrix",
    "_init_imm_pair_log_alpha",
    "_init_pair_log_alpha",
    "_mean_entropy",
    "_mode_transition_matrix",
    "_normalize_log_weights",
    "_pairwise_gaussian_log_prob",
    "_per_bin_sigma",
    "_posterior_diagnostics",
    "_scaled_emissions",
    "_score_first_order_imm",
    "_score_fragmented",
    "_score_imm_candidates",
    "_score_momentum_candidates",
    "_score_stationary",
    "_top_candidate_indices",
    "_validate_candidate_indices",
]
