"""State-space replay decoder baselines with full trajectory posteriors."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .encoding import LogEmissionTensor
from .models import EventScore, _posterior_diagnostics
from .state_space_candidates import _score_imm_candidates
from .state_space_candidates_momentum import _score_momentum_candidates
from .state_space_first_order import (
    _forward_backward_first_order,
    _score_first_order_imm,
    _score_fragmented,
    _score_stationary,
)
from .state_space_utils import (
    _gaussian_transition_matrix,
    _mean_entropy,
    _per_bin_sigma,
    _top_candidate_indices,
    _validate_candidate_indices,
)


@dataclass(frozen=True)
class StateSpaceDecoderConfig:
    """Configuration for state-space replay baselines.

    Diffusion and momentum noise are specified in cm/sqrt(s). They are converted
    to per-bin standard deviations using the emission tensor's ``dt`` so that the
    same model can be evaluated with 1--3 ms replay bins.
    """

    mode: str = "diffusion"
    stationary_sigma_cm: float = 2.0
    diffusion_sigma_cm_sqrt_s: float = 85.0
    max_step_sigma: float = 4.0
    imm_mode_stickiness: float = 0.95
    momentum_sigma_cm_sqrt_s: float = 85.0
    momentum_initial_sigma_cm_sqrt_s: float = 85.0
    momentum_velocity_decay: float = 0.95
    momentum_candidate_top_k: int = 128
    momentum_predicted_candidate_top_k: int = 8


@dataclass
class StateSpaceReplayModel:
    """Replay decoder baseline returning a posterior for every replay bin.

    Supported modes are ``stationary``, ``diffusion``, ``fragmented``/``jump``,
    ``first-order-imm``, ``momentum``, and ``imm``. The first-order models use
    exact full-grid forward-backward recursions. ``first-order-imm`` is the
    legacy exact first-order switcher over stationary, diffusion, and
    fragmented/jump dynamics. ``momentum`` and ``imm`` use candidate-pruned
    second-order dynamics for scalability. ``imm`` switches among stationary,
    diffusion, momentum, and jump modes.

    Candidate recursions keep full-grid prior and transition normalizers and
    drop off-support paths, so their evidences are conservative truncated
    full-grid evidences. They return candidate-supported per-bin posterior
    marginals.
    """

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

    def candidate_indices(self, emissions: LogEmissionTensor, bin_centers: np.ndarray | None = None) -> list[np.ndarray]:
        """Return the candidate support used by pruned momentum/IMM recursions.

        The base support is the per-bin emission top-k set. When
        ``momentum_predicted_candidate_top_k`` is positive and ``bin_centers`` is
        supplied, each time bin is enlarged by nearest grid states to forward-
        and backward-momentum predictions built from the top adjacent emission
        candidates. This keeps deterministic emission support while recovering
        dynamically plausible states whose local emission rank is too low for
        the fixed top-k beam. The augmentation is bounded by the prediction top-k
        and never changes externally supplied candidate supports.
        """

        assert self.config is not None
        base = [_top_candidate_indices(row, self.config.momentum_candidate_top_k) for row in emissions.log_likelihood]
        predicted_top_k = int(self.config.momentum_predicted_candidate_top_k)
        if predicted_top_k <= 0 or bin_centers is None or emissions.n_time < 3:
            return base
        return _augment_candidates_with_momentum_predictions(
            base,
            bin_centers,
            predicted_top_k=predicted_top_k,
            velocity_decay=float(self.config.momentum_velocity_decay),
        )

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
            momentum_initial_sigma_cm = _per_bin_sigma(
                self.config.momentum_initial_sigma_cm_sqrt_s,
                emissions.dt,
            )
            candidates = self.candidate_indices(emissions, bin_centers) if candidate_indices is None else candidate_indices
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
                    "mean_candidate_count": float(np.mean([len(curr) for curr in candidates])),
                    "state_space_imm_modes": ",".join(names),
                    "state_space_imm_candidate_top_k": int(self.config.momentum_candidate_top_k),
                    "state_space_imm_predicted_candidate_top_k": int(self.config.momentum_predicted_candidate_top_k),
                    "state_space_imm_candidate_support": "derived" if candidate_indices is None else "provided",
                    "state_space_imm_trajectory_posterior": "smoothed_pair_marginal",
                    "state_space_imm_evidence_support": "truncated_full_grid",
                    "state_space_momentum_transition_sigma_cm": float(momentum_transition_sigma_cm),
                    "state_space_momentum_initial_transition_sigma_cm": float(momentum_initial_sigma_cm),
                }
            )
        elif self.mode == "momentum":
            transition_sigma_cm = _per_bin_sigma(self.config.momentum_sigma_cm_sqrt_s, emissions.dt)
            candidates = self.candidate_indices(emissions, bin_centers) if candidate_indices is None else candidate_indices
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
                "mean_candidate_count": float(np.mean([len(curr) for curr in candidates])),
                "state_space_momentum_candidate_top_k": int(self.config.momentum_candidate_top_k),
                "state_space_momentum_predicted_candidate_top_k": int(self.config.momentum_predicted_candidate_top_k),
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


def _augment_candidates_with_momentum_predictions(
    candidates: list[np.ndarray],
    bin_centers: np.ndarray,
    *,
    predicted_top_k: int,
    velocity_decay: float,
) -> list[np.ndarray]:
    """Union emission candidates with states nearest to bounded momentum predictions."""

    if predicted_top_k <= 0:
        return [np.asarray(curr, dtype=int).copy() for curr in candidates]
    top_k = max(1, int(predicted_top_k))
    augmented = [set(int(idx) for idx in np.asarray(curr, dtype=int)) for curr in candidates]

    for time_index in range(2, len(candidates)):
        prev_prev = np.asarray(candidates[time_index - 2], dtype=int)[:top_k]
        prev = np.asarray(candidates[time_index - 1], dtype=int)[:top_k]
        if prev_prev.size == 0 or prev.size == 0:
            continue
        predictions = bin_centers[prev][None, :, :] + velocity_decay * (
            bin_centers[prev][None, :, :] - bin_centers[prev_prev][:, None, :]
        )
        _add_nearest_predictions(augmented[time_index], bin_centers, predictions)

    if abs(velocity_decay) > np.finfo(float).eps:
        for time_index in range(len(candidates) - 2):
            nxt = np.asarray(candidates[time_index + 1], dtype=int)[:top_k]
            nxt_nxt = np.asarray(candidates[time_index + 2], dtype=int)[:top_k]
            if nxt.size == 0 or nxt_nxt.size == 0:
                continue
            predictions = bin_centers[nxt][None, :, :] - (
                bin_centers[nxt_nxt][:, None, :] - bin_centers[nxt][None, :, :]
            ) / velocity_decay
            _add_nearest_predictions(augmented[time_index], bin_centers, predictions)

    return [np.fromiter(sorted(curr), dtype=int) for curr in augmented]


def _add_nearest_predictions(
    target: set[int],
    bin_centers: np.ndarray,
    predictions: np.ndarray,
) -> None:
    flat = predictions.reshape(-1, bin_centers.shape[1])
    if flat.size == 0:
        return
    for predicted in flat:
        dist2 = np.sum((bin_centers - predicted[None, :]) ** 2, axis=1)
        target.add(int(np.argmin(dist2)))
