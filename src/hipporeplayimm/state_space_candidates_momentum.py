"""Candidate-pruned momentum replay recursion."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import LOG_ZERO
from .state_space_first_order import _score_fragmented
from .state_space_utils import (
    _candidate_log_masses,
    _full_grid_normalized_pairwise_gaussian_log_prob,
)


def _score_momentum_candidates(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    candidates: list[np.ndarray],
    *,
    sigma_cm: float | np.ndarray,
    initial_sigma_cm: float,
    velocity_decay: float | np.ndarray,
    time_scales: float | np.ndarray = 1.0,
) -> tuple[float, np.ndarray, list[float]]:
    if emissions.n_time == 1:
        logp, trajectory = _score_fragmented(emissions)
        return logp, trajectory, [0.0]

    masses = _candidate_log_masses(emissions.log_likelihood, candidates)
    log_pair = _init_pair_log_alpha(
        emissions.log_likelihood,
        candidates[0],
        candidates[1],
        bin_centers,
        sigma_cm=initial_sigma_cm,
    )
    pair_alphas = [log_pair]
    for time_index in range(2, emissions.n_time):
        transition_index = time_index - 1
        log_pair = _advance_momentum_pair(
            log_pair,
            candidates[time_index - 2],
            candidates[time_index - 1],
            candidates[time_index],
            emissions.log_likelihood[time_index, candidates[time_index]],
            bin_centers,
            sigma_cm=_value_at(sigma_cm, transition_index),
            velocity_decay=_value_at(velocity_decay, transition_index),
            time_scale=_value_at(time_scales, transition_index),
        )
        pair_alphas.append(log_pair)

    logp = float(logsumexp(pair_alphas[-1]))
    pair_betas = [np.zeros_like(pair_alphas[-1]) for _ in pair_alphas]
    for pair_index in range(len(pair_alphas) - 2, -1, -1):
        transition_index = pair_index + 1
        curr_time = pair_index + 2
        pair_betas[pair_index] = _backward_momentum_pair(
            pair_betas[pair_index + 1],
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[curr_time],
            emissions.log_likelihood[curr_time, candidates[curr_time]],
            bin_centers,
            sigma_cm=_value_at(sigma_cm, transition_index),
            velocity_decay=_value_at(velocity_decay, transition_index),
            time_scale=_value_at(time_scales, transition_index),
        )

    trajectory = np.full((emissions.n_time, emissions.n_bins), LOG_ZERO, dtype=float)
    for pair_index, (alpha, beta) in enumerate(zip(pair_alphas, pair_betas, strict=True)):
        pair_log_posterior = alpha + beta - logp
        if pair_index == 0:
            trajectory[0, candidates[0]] = logsumexp(pair_log_posterior, axis=1)
        trajectory[pair_index + 1, candidates[pair_index + 1]] = logsumexp(pair_log_posterior, axis=0)
    for time_index in range(emissions.n_time):
        trajectory[time_index] -= logsumexp(trajectory[time_index])
    return logp, trajectory, masses


def _init_pair_log_alpha(
    log_likelihood: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    bin_centers: np.ndarray,
    *,
    sigma_cm: float,
) -> np.ndarray:
    log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
        bin_centers[first],
        bin_centers[second],
        bin_centers,
        sigma_cm,
    )
    n_bins = log_likelihood.shape[1]
    return log_likelihood[0, first][:, None] - np.log(n_bins) + log_kernel + log_likelihood[1, second][None, :]


def _advance_momentum_pair(
    log_pair: np.ndarray,
    prev_prev: np.ndarray,
    prev: np.ndarray,
    curr: np.ndarray,
    curr_emission: np.ndarray,
    bin_centers: np.ndarray,
    *,
    sigma_cm: float,
    velocity_decay: float,
    time_scale: float = 1.0,
) -> np.ndarray:
    coords_prev_prev = bin_centers[prev_prev]
    coords_prev = bin_centers[prev]
    coords_curr = bin_centers[curr]
    output = np.full((len(prev), len(curr)), LOG_ZERO, dtype=float)
    coefficient = float(velocity_decay) * float(time_scale)
    for prev_col in range(len(prev)):
        predictions = coords_prev[prev_col][None, :] + coefficient * (
            coords_prev[prev_col][None, :] - coords_prev_prev
        )
        log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
            predictions,
            coords_curr,
            bin_centers,
            sigma_cm,
        )
        output[prev_col] = logsumexp(log_pair[:, prev_col][:, None] + log_kernel, axis=0) + curr_emission
    return output


def _backward_momentum_pair(
    next_beta: np.ndarray,
    prev_prev: np.ndarray,
    prev: np.ndarray,
    curr: np.ndarray,
    curr_emission: np.ndarray,
    bin_centers: np.ndarray,
    *,
    sigma_cm: float,
    velocity_decay: float,
    time_scale: float = 1.0,
) -> np.ndarray:
    coords_prev_prev = bin_centers[prev_prev]
    coords_prev = bin_centers[prev]
    coords_curr = bin_centers[curr]
    output = np.full((len(prev_prev), len(prev)), LOG_ZERO, dtype=float)
    coefficient = float(velocity_decay) * float(time_scale)
    for prev_col in range(len(prev)):
        predictions = coords_prev[prev_col][None, :] + coefficient * (
            coords_prev[prev_col][None, :] - coords_prev_prev
        )
        log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
            predictions,
            coords_curr,
            bin_centers,
            sigma_cm,
        )
        continuation = curr_emission[None, :] + next_beta[prev_col][None, :]
        output[:, prev_col] = logsumexp(log_kernel + continuation, axis=1)
    return output


def _value_at(value: float | np.ndarray, index: int) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return float(array[index])
