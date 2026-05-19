"""Duration-aware four-mode state-space IMM patch."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .duration_dynamics import (
    _decays,
    _ps,
    _pss,
    _rep,
    _scales,
    attach_duration_metadata,
    transition_durations_s,
)


def apply_state_space_imm_duration_patch() -> None:
    """Deprecated compatibility shim.

    Four-mode IMM duration scaling is implemented directly in
    ``StateSpaceReplayModel.score`` and ``_score_imm_candidates``. This function
    is kept so older scripts that import it do not fail, but it intentionally no
    longer mutates the public state-space model.
    """
    return None


def _score_imm_duration(
    ss,
    emissions,
    bin_centers,
    candidates,
    *,
    stationary_sigma_cm,
    diffusion_sigmas_cm,
    momentum_sigmas_cm,
    initial_momentum_sigma_cm,
    velocity_decays,
    time_scales,
    mode_stickiness,
):
    """Candidate-pruned four-mode IMM with duration-dependent transitions."""

    modes = ("stationary", "diffusion", "momentum", "jump")
    if emissions.n_time == 1:
        logp, trajectory = ss._score_fragmented(emissions)
        mode_posterior = np.full((1, len(modes)), 1.0 / len(modes), dtype=float)
        return logp, trajectory, mode_posterior, [0.0]

    masses = ss._candidate_log_masses(emissions.log_likelihood, candidates)
    mode_transition = ss._mode_transition_matrix(len(modes), mode_stickiness)
    with np.errstate(divide="ignore"):
        log_mode_transition = np.log(mode_transition)

    by_mode = [
        ss._init_imm_pair_log_alpha(
            emissions.log_likelihood,
            candidates[0],
            candidates[1],
            bin_centers,
            mode=mode,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigma_cm=float(diffusion_sigmas_cm[0]),
            momentum_initial_sigma_cm=float(initial_momentum_sigma_cm),
        )
        for mode in modes
    ]
    log_pair = np.stack(by_mode, axis=0) - np.log(len(modes))
    pair_alphas = [log_pair]

    for time_index in range(2, emissions.n_time):
        transition_index = time_index - 1
        next_alpha = []
        for dst_mode_index, dst_mode in enumerate(modes):
            mixed_prev = logsumexp(
                log_pair + log_mode_transition[:, dst_mode_index][:, None, None],
                axis=0,
            )
            next_alpha.append(
                ss._advance_imm_pair_log_alpha(
                    mixed_prev,
                    candidates[time_index - 2],
                    candidates[time_index - 1],
                    candidates[time_index],
                    emissions.log_likelihood[time_index, candidates[time_index]],
                    bin_centers,
                    mode=dst_mode,
                    stationary_sigma_cm=stationary_sigma_cm,
                    diffusion_sigma_cm=float(diffusion_sigmas_cm[transition_index]),
                    momentum_sigma_cm=float(momentum_sigmas_cm[transition_index]),
                    velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
                )
            )
        log_pair = np.stack(next_alpha, axis=0)
        pair_alphas.append(log_pair)

    logp = float(logsumexp(pair_alphas[-1]))
    pair_betas = [np.zeros_like(pair_alphas[-1]) for _ in pair_alphas]
    for pair_index in range(len(pair_alphas) - 2, -1, -1):
        transition_index = pair_index + 1
        curr_time = pair_index + 2
        pair_betas[pair_index] = ss._backward_imm_pair(
            pair_betas[pair_index + 1],
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[curr_time],
            emissions.log_likelihood[curr_time, candidates[curr_time]],
            bin_centers,
            modes=modes,
            mode_transition=mode_transition,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigma_cm=float(diffusion_sigmas_cm[transition_index]),
            momentum_sigma_cm=float(momentum_sigmas_cm[transition_index]),
            velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
        )

    trajectory = np.full((emissions.n_time, emissions.n_bins), ss.LOG_ZERO, dtype=float)
    mode_log_posterior = np.full((emissions.n_time, len(modes)), ss.LOG_ZERO, dtype=float)
    for pair_index, (alpha, beta) in enumerate(zip(pair_alphas, pair_betas, strict=True)):
        pair_log_posterior = alpha + beta - logp
        if pair_index == 0:
            trajectory[0, candidates[0]] = logsumexp(pair_log_posterior, axis=(0, 2))
            mode_log_posterior[0] = logsumexp(pair_log_posterior, axis=(1, 2))
        trajectory[pair_index + 1, candidates[pair_index + 1]] = logsumexp(pair_log_posterior, axis=(0, 1))
        mode_log_posterior[pair_index + 1] = logsumexp(pair_log_posterior, axis=(1, 2))
    for time_index in range(emissions.n_time):
        trajectory[time_index] -= logsumexp(trajectory[time_index])
    mode_log_posterior -= logsumexp(mode_log_posterior, axis=1)[:, None]
    return logp, trajectory, np.exp(mode_log_posterior), masses
