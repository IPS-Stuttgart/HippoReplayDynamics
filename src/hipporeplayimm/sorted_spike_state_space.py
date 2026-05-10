"""Sorted-spike state-space replay decoder labels.

The current state-space replay baseline consumes Poisson emissions built from
sorted spike identities. This wrapper makes that observation model explicit in
model names and diagnostics so it is not confused with a true clusterless marked
point-process decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import EventScore, LOG_ZERO, _posterior_diagnostics
from .state_space import (
    StateSpaceDecoderConfig,
    StateSpaceReplayModel,
    _advance_momentum_pair,
    _as_log_probs,
    _candidate_log_masses,
    _init_pair_log_alpha,
    _mean_entropy,
    _per_bin_sigma,
    _score_fragmented,
    _top_candidate_indices,
)


@dataclass
class SortedSpikeStateSpaceReplayModel(StateSpaceReplayModel):
    """State-space replay model using sorted-spike Poisson emissions."""

    mode: str = "diffusion"
    config: StateSpaceDecoderConfig | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name is None or self.name.startswith("state-space-"):
            self.name = f"sorted-spike-state-space-{self.mode}"

    def candidate_indices(self, emissions: LogEmissionTensor) -> list[np.ndarray]:
        """Return momentum candidate support for an emission tensor.

        The support is defined over position-grid indices only. Held-out scoring
        can therefore select it from train-cell emissions and reuse the exact same
        support for joint train+test emissions.
        """

        if self.mode != "momentum":
            raise ValueError("candidate_indices is only defined for momentum mode")
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        assert self.config is not None
        return [
            _top_candidate_indices(row, self.config.momentum_candidate_top_k)
            for row in emissions.log_likelihood
        ]

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray] | None = None,
    ) -> EventScore:
        if candidate_indices is not None:
            if self.mode != "momentum":
                raise ValueError("candidate_indices is only supported for momentum mode")
            score = self._score_momentum_with_fixed_candidates(
                emissions,
                bin_centers,
                candidate_indices,
            )
        else:
            score = super().score(emissions, bin_centers)
        score.model_name = str(self.name)
        score.diagnostics["state_space_observation_model"] = "sorted-spike-poisson"
        score.diagnostics["clusterless_mark_likelihood"] = "not_implemented"
        return score

    def _score_momentum_with_fixed_candidates(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray],
    ) -> EventScore:
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        if emissions.n_bins != bin_centers.shape[0]:
            raise ValueError("emissions.n_bins must match bin_centers rows")
        assert self.config is not None
        transition_sigma_cm = _per_bin_sigma(
            self.config.momentum_sigma_cm_sqrt_s,
            emissions.dt,
        )
        logp, trajectory, masses = _score_momentum_with_candidates(
            emissions,
            bin_centers,
            candidate_indices,
            sigma_cm=transition_sigma_cm,
            initial_sigma_cm=_per_bin_sigma(
                self.config.momentum_initial_sigma_cm_sqrt_s,
                emissions.dt,
            ),
            velocity_decay=self.config.momentum_velocity_decay,
        )
        terminal = trajectory[-1]
        diagnostics: dict[str, float | int | str] = {
            "state_space_mode": str(self.mode),
            "state_space_time_bin_s": float(emissions.dt),
            "state_space_trajectory_posterior": 1,
            "state_space_trajectory_time_bins": int(emissions.n_time),
            "state_space_transition_sigma_cm": float(transition_sigma_cm),
            "mean_trajectory_posterior_entropy": _mean_entropy(trajectory),
            "mean_candidate_log_mass": float(np.mean(masses)),
            "state_space_momentum_candidate_top_k": int(
                self.config.momentum_candidate_top_k
            ),
            "state_space_fixed_candidate_support": 1,
            "state_space_momentum_candidate_support": "provided",
            "state_space_momentum_trajectory_posterior": "smoothed_pair_marginal",
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


def _score_momentum_with_candidates(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    candidate_indices: list[np.ndarray],
    *,
    sigma_cm: float,
    initial_sigma_cm: float,
    velocity_decay: float,
) -> tuple[float, np.ndarray, list[float]]:
    if emissions.n_time == 1:
        logp, trajectory = _score_fragmented(emissions)
        return logp, trajectory, [0.0]

    candidates = _validate_candidate_indices(
        candidate_indices,
        n_time=emissions.n_time,
        n_bins=emissions.n_bins,
    )
    masses = _candidate_log_masses(emissions.log_likelihood, candidates)
    log_pair = _init_pair_log_alpha(
        emissions.log_likelihood,
        candidates[0],
        candidates[1],
        bin_centers,
        sigma_cm=initial_sigma_cm,
    )
    trajectory = np.full((emissions.n_time, emissions.n_bins), LOG_ZERO, dtype=float)
    first_pair = log_pair - logsumexp(log_pair)
    trajectory[0, candidates[0]] = logsumexp(first_pair, axis=1)
    trajectory[1, candidates[1]] = logsumexp(first_pair, axis=0)

    for time_index in range(2, emissions.n_time):
        log_pair = _advance_momentum_pair(
            log_pair,
            candidates[time_index - 2],
            candidates[time_index - 1],
            candidates[time_index],
            emissions.log_likelihood[time_index, candidates[time_index]],
            bin_centers,
            sigma_cm=sigma_cm,
            velocity_decay=velocity_decay,
        )
        current_pair = log_pair - logsumexp(log_pair)
        trajectory[time_index, candidates[time_index]] = logsumexp(
            current_pair,
            axis=0,
        )

    return float(logsumexp(log_pair)), _as_log_probs(np.exp(trajectory)), masses


def _validate_candidate_indices(
    candidate_indices: list[np.ndarray],
    *,
    n_time: int,
    n_bins: int,
) -> list[np.ndarray]:
    if len(candidate_indices) != n_time:
        raise ValueError("candidate_indices must contain one array per emission time bin")
    validated: list[np.ndarray] = []
    for time_index, indices in enumerate(candidate_indices):
        arr = np.asarray(indices, dtype=int).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"candidate_indices[{time_index}] must not be empty")
        if np.any(arr < 0) or np.any(arr >= n_bins):
            raise ValueError(
                f"candidate_indices[{time_index}] contains grid indices outside 0..{n_bins - 1}"
            )
        if arr.size > 1:
            _, first_positions = np.unique(arr, return_index=True)
            if first_positions.size != arr.size:
                arr = arr[np.sort(first_positions)]
        validated.append(arr)
    return validated
