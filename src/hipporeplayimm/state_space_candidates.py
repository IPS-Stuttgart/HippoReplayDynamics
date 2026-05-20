"""Candidate-pruned IMM replay recursion."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import LOG_ZERO
from .state_space_first_order import _score_fragmented
from .state_space_candidates_momentum import _backward_momentum_pair
from .state_space_utils import (
    _candidate_log_masses,
    _full_grid_normalized_pairwise_gaussian_log_prob,
    _uniform_log_prior,
    _valid_bin_count,
)


def _score_imm_candidates(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    *,
    stationary_sigma_cm: float,
    diffusion_sigma_cm: float,
    momentum_sigma_cm: float,
    momentum_initial_sigma_cm: float,
    velocity_decay: float,
    mode_stickiness: float,
    candidate_indices: list[np.ndarray],
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    """Candidate-pruned four-mode IMM over stationary/diffusion/momentum/jump."""

    modes = ("stationary", "diffusion", "momentum", "jump")
    if emissions.n_time == 1:
        logp, trajectory = _score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
        mode_post = np.full((1, len(modes)), 1.0 / len(modes), dtype=float)
        return logp, trajectory, mode_post, [0.0]

    mode_transition = _mode_transition_matrix(len(modes), mode_stickiness)
    with np.errstate(divide="ignore"):
        log_mode_transition = np.log(mode_transition)

    masses = _candidate_log_masses(emissions.log_likelihood, candidate_indices)
    first = candidate_indices[0]
    second = candidate_indices[1]
    by_mode = [
        _init_imm_pair_log_alpha(
            emissions.log_likelihood,
            first,
            second,
            bin_centers,
            mode=mode,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigma_cm=diffusion_sigma_cm,
            momentum_initial_sigma_cm=momentum_initial_sigma_cm,
            valid_bin_mask=valid_bin_mask,
        )
        for mode in modes
    ]
    log_pair = np.stack(by_mode, axis=0) - np.log(len(modes))
    pair_alphas = [log_pair]

    for time_index in range(2, emissions.n_time):
        prev_prev = candidate_indices[time_index - 2]
        prev = candidate_indices[time_index - 1]
        curr = candidate_indices[time_index]
        next_alpha = []
        for dst_mode_index, dst_mode in enumerate(modes):
            mixed_prev = logsumexp(
                log_pair + log_mode_transition[:, dst_mode_index][:, None, None],
                axis=0,
            )
            next_alpha.append(
                _advance_imm_pair_log_alpha(
                    mixed_prev,
                    prev_prev,
                    prev,
                    curr,
                    emissions.log_likelihood[time_index, curr],
                    bin_centers,
                    mode=dst_mode,
                    stationary_sigma_cm=stationary_sigma_cm,
                    diffusion_sigma_cm=diffusion_sigma_cm,
                    momentum_sigma_cm=momentum_sigma_cm,
                    velocity_decay=velocity_decay,
                    valid_bin_mask=valid_bin_mask,
                )
            )
        log_pair = np.stack(next_alpha, axis=0)
        pair_alphas.append(log_pair)

    logp = float(logsumexp(pair_alphas[-1]))
    pair_betas = [np.zeros_like(pair_alphas[-1]) for _ in pair_alphas]
    for pair_index in range(len(pair_alphas) - 2, -1, -1):
        curr_time = pair_index + 2
        pair_betas[pair_index] = _backward_imm_pair(
            pair_betas[pair_index + 1],
            candidate_indices[pair_index],
            candidate_indices[pair_index + 1],
            candidate_indices[curr_time],
            emissions.log_likelihood[curr_time, candidate_indices[curr_time]],
            bin_centers,
            modes=modes,
            mode_transition=mode_transition,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigma_cm=diffusion_sigma_cm,
            momentum_sigma_cm=momentum_sigma_cm,
            velocity_decay=velocity_decay,
            valid_bin_mask=valid_bin_mask,
        )

    trajectory = np.full((emissions.n_time, emissions.n_bins), LOG_ZERO, dtype=float)
    mode_log_posterior = np.full((emissions.n_time, len(modes)), LOG_ZERO, dtype=float)
    for pair_index, (alpha, beta) in enumerate(zip(pair_alphas, pair_betas, strict=True)):
        pair_log_posterior = alpha + beta - logp
        if pair_index == 0:
            trajectory[0, candidate_indices[0]] = logsumexp(pair_log_posterior, axis=(0, 2))
            mode_log_posterior[0] = logsumexp(pair_log_posterior, axis=(1, 2))
        trajectory[pair_index + 1, candidate_indices[pair_index + 1]] = logsumexp(
            pair_log_posterior,
            axis=(0, 1),
        )
        mode_log_posterior[pair_index + 1] = logsumexp(pair_log_posterior, axis=(1, 2))
    for time_index in range(emissions.n_time):
        trajectory[time_index] -= logsumexp(trajectory[time_index])
    mode_log_posterior -= logsumexp(mode_log_posterior, axis=1)[:, None]
    return logp, trajectory, np.exp(mode_log_posterior), masses


def _init_imm_pair_log_alpha(
    log_likelihood: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    bin_centers: np.ndarray,
    *,
    mode: str,
    stationary_sigma_cm: float,
    diffusion_sigma_cm: float,
    momentum_initial_sigma_cm: float,
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    n_bins = log_likelihood.shape[1]
    if mode == "jump":
        log_kernel = np.full(
            (len(first), len(second)),
            -np.log(_valid_bin_count(n_bins, valid_bin_mask)),
            dtype=float,
        )
    else:
        if mode == "stationary":
            sigma_cm = stationary_sigma_cm
        elif mode == "diffusion":
            sigma_cm = diffusion_sigma_cm
        elif mode == "momentum":
            sigma_cm = momentum_initial_sigma_cm
        else:
            raise ValueError(f"Unknown IMM mode: {mode}")
        log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
            bin_centers[first],
            bin_centers[second],
            bin_centers,
            sigma_cm,
            valid_bin_mask=valid_bin_mask,
        )
    return (
        log_likelihood[0, first][:, None]
        + _uniform_log_prior(n_bins, valid_bin_mask)[first][:, None]
        + log_kernel
        + log_likelihood[1, second][None, :]
    )


def _advance_imm_pair_log_alpha(
    log_pair: np.ndarray,
    prev_prev: np.ndarray,
    prev: np.ndarray,
    curr: np.ndarray,
    curr_emission: np.ndarray,
    bin_centers: np.ndarray,
    *,
    mode: str,
    stationary_sigma_cm: float,
    diffusion_sigma_cm: float,
    momentum_sigma_cm: float,
    velocity_decay: float,
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    coords_prev_prev = bin_centers[prev_prev]
    coords_prev = bin_centers[prev]
    coords_curr = bin_centers[curr]
    output = np.full((len(prev), len(curr)), LOG_ZERO, dtype=float)
    if mode == "jump":
        collapsed_by_prev = logsumexp(log_pair, axis=0)
        return (
            collapsed_by_prev[:, None]
            - np.log(_valid_bin_count(bin_centers.shape[0], valid_bin_mask))
            + curr_emission[None, :]
        )
    for prev_col in range(len(prev)):
        if mode == "stationary":
            previous_mass = logsumexp(log_pair[:, prev_col])
            log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
                coords_prev[prev_col][None, :],
                coords_curr,
                bin_centers,
                stationary_sigma_cm,
                valid_bin_mask=valid_bin_mask,
            )[0]
            output[prev_col] = previous_mass + log_kernel + curr_emission
        elif mode == "diffusion":
            previous_mass = logsumexp(log_pair[:, prev_col])
            log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
                coords_prev[prev_col][None, :],
                coords_curr,
                bin_centers,
                diffusion_sigma_cm,
                valid_bin_mask=valid_bin_mask,
            )[0]
            output[prev_col] = previous_mass + log_kernel + curr_emission
        elif mode == "momentum":
            predictions = coords_prev[prev_col][None, :] + velocity_decay * (
                coords_prev[prev_col][None, :] - coords_prev_prev
            )
            log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
                predictions,
                coords_curr,
                bin_centers,
                momentum_sigma_cm,
                valid_bin_mask=valid_bin_mask,
            )
            output[prev_col] = logsumexp(log_pair[:, prev_col][:, None] + log_kernel, axis=0) + curr_emission
        else:
            raise ValueError(f"Unknown IMM mode: {mode}")
    return output


def _backward_imm_pair(
    next_beta: np.ndarray,
    prev_prev: np.ndarray,
    prev: np.ndarray,
    curr: np.ndarray,
    curr_emission: np.ndarray,
    bin_centers: np.ndarray,
    *,
    modes: tuple[str, ...],
    mode_transition: np.ndarray,
    stationary_sigma_cm: float,
    diffusion_sigma_cm: float,
    momentum_sigma_cm: float,
    velocity_decay: float,
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    dst_terms = np.stack(
        [
            _backward_imm_pair_for_mode(
                next_beta[dst_idx],
                prev_prev,
                prev,
                curr,
                curr_emission,
                bin_centers,
                mode=dst_mode,
                stationary_sigma_cm=stationary_sigma_cm,
                diffusion_sigma_cm=diffusion_sigma_cm,
                momentum_sigma_cm=momentum_sigma_cm,
                velocity_decay=velocity_decay,
                valid_bin_mask=valid_bin_mask,
            )
            for dst_idx, dst_mode in enumerate(modes)
        ],
        axis=0,
    )
    with np.errstate(divide="ignore"):
        log_mode_transition = np.log(mode_transition)
    output = np.full((len(modes), len(prev_prev), len(prev)), LOG_ZERO, dtype=float)
    for src_idx in range(len(modes)):
        output[src_idx] = logsumexp(dst_terms + log_mode_transition[src_idx, :, None, None], axis=0)
    return output


def _backward_imm_pair_for_mode(
    next_beta: np.ndarray,
    prev_prev: np.ndarray,
    prev: np.ndarray,
    curr: np.ndarray,
    curr_emission: np.ndarray,
    bin_centers: np.ndarray,
    *,
    mode: str,
    stationary_sigma_cm: float,
    diffusion_sigma_cm: float,
    momentum_sigma_cm: float,
    velocity_decay: float,
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "momentum":
        return _backward_momentum_pair(
            next_beta,
            prev_prev,
            prev,
            curr,
            curr_emission,
            bin_centers,
            sigma_cm=momentum_sigma_cm,
            velocity_decay=velocity_decay,
            valid_bin_mask=valid_bin_mask,
        )

    output = np.full((len(prev_prev), len(prev)), LOG_ZERO, dtype=float)
    if mode == "jump":
        values_by_prev = logsumexp(
            -np.log(_valid_bin_count(bin_centers.shape[0], valid_bin_mask))
            + curr_emission[None, :]
            + next_beta,
            axis=1,
        )
        output[:, :] = values_by_prev[None, :]
        return output

    if mode == "stationary":
        sigma_cm = stationary_sigma_cm
    elif mode == "diffusion":
        sigma_cm = diffusion_sigma_cm
    else:
        raise ValueError(f"Unknown IMM mode: {mode}")

    coords_prev = bin_centers[prev]
    coords_curr = bin_centers[curr]
    for prev_col in range(len(prev)):
        log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
            coords_prev[prev_col][None, :],
            coords_curr,
            bin_centers,
            sigma_cm,
            valid_bin_mask=valid_bin_mask,
        )[0]
        output[:, prev_col] = logsumexp(log_kernel + curr_emission + next_beta[prev_col])
    return output


def _mode_transition_matrix(n_modes: int, stickiness: float) -> np.ndarray:
    if n_modes < 2:
        return np.ones((n_modes, n_modes), dtype=float)
    if not 0.0 <= stickiness <= 1.0:
        raise ValueError("mode_stickiness must be in [0, 1]")
    off_diag = (1.0 - stickiness) / (n_modes - 1)
    matrix = np.full((n_modes, n_modes), off_diag, dtype=float)
    np.fill_diagonal(matrix, stickiness)
    return matrix
