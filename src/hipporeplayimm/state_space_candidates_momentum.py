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
    _uniform_log_prior,
)


def _score_momentum_candidates(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    candidates: list[np.ndarray],
    *,
    sigma_cm: float,
    initial_sigma_cm: float,
    velocity_decay: float,
    transition_sigmas_cm: np.ndarray | None = None,
    velocity_decays: np.ndarray | None = None,
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, list[float]]:
    if emissions.n_time == 1:
        logp, trajectory = _score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
        return logp, trajectory, [0.0]

    masses = _candidate_log_masses(emissions.log_likelihood, candidates)
    transition_sigmas = _transition_parameter_series(
        transition_sigmas_cm,
        emissions.n_time - 1,
        sigma_cm,
        name="transition_sigmas_cm",
        minimum=0.0,
        include_minimum=False,
    )
    transition_velocity_decays = _transition_parameter_series(
        velocity_decays,
        emissions.n_time - 1,
        velocity_decay,
        name="velocity_decays",
        minimum=0.0,
        maximum=1.0,
    )
    log_pair = _init_pair_log_alpha(
        emissions.log_likelihood,
        candidates[0],
        candidates[1],
        bin_centers,
        sigma_cm=initial_sigma_cm,
        valid_bin_mask=valid_bin_mask,
    )
    pair_alphas = [log_pair]
    for time_index in range(2, emissions.n_time):
        log_pair = _advance_momentum_pair(
            log_pair,
            candidates[time_index - 2],
            candidates[time_index - 1],
            candidates[time_index],
            emissions.log_likelihood[time_index, candidates[time_index]],
            bin_centers,
            sigma_cm=float(transition_sigmas[time_index - 1]),
            velocity_decay=float(transition_velocity_decays[time_index - 1]),
            valid_bin_mask=valid_bin_mask,
        )
        pair_alphas.append(log_pair)

    logp = float(logsumexp(pair_alphas[-1]))
    pair_betas = [np.zeros_like(pair_alphas[-1]) for _ in pair_alphas]
    for pair_index in range(len(pair_alphas) - 2, -1, -1):
        curr_time = pair_index + 2
        pair_betas[pair_index] = _backward_momentum_pair(
            pair_betas[pair_index + 1],
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[curr_time],
            emissions.log_likelihood[curr_time, candidates[curr_time]],
            bin_centers,
            sigma_cm=float(transition_sigmas[curr_time - 1]),
            velocity_decay=float(transition_velocity_decays[curr_time - 1]),
            valid_bin_mask=valid_bin_mask,
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


def _transition_parameter_series(
    values: np.ndarray | None,
    n_transitions: int,
    fallback: float,
    *,
    name: str,
    minimum: float | None = None,
    include_minimum: bool = True,
    maximum: float | None = None,
    include_maximum: bool = True,
) -> np.ndarray:
    """Return one scalar transition parameter per adjacent time-bin pair."""

    if n_transitions <= 0:
        return np.empty(0, dtype=float)
    if values is None:
        out = np.full(n_transitions, float(fallback), dtype=float)
    else:
        out = np.asarray(values, dtype=float)
        if out.shape != (n_transitions,):
            raise ValueError(f"{name} must contain one value per transition")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        minimum_value = float(minimum)
        if not np.isfinite(minimum_value):
            raise ValueError("minimum must be finite")
        if include_minimum:
            if np.any(out < minimum_value):
                raise ValueError(f"{name} values must be >= {minimum_value:g}")
        elif np.any(out <= minimum_value):
            raise ValueError(f"{name} values must be > {minimum_value:g}")
    if maximum is None and name == "velocity_decays":
        maximum = 1.0
    if maximum is not None:
        maximum_value = float(maximum)
        if not np.isfinite(maximum_value):
            raise ValueError("maximum must be finite")
        if include_maximum:
            if np.any(out > maximum_value):
                raise ValueError(f"{name} values must be <= {maximum_value:g}")
        elif np.any(out >= maximum_value):
            raise ValueError(f"{name} values must be < {maximum_value:g}")
    return out


def _init_pair_log_alpha(
    log_likelihood: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    bin_centers: np.ndarray,
    *,
    sigma_cm: float,
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
        bin_centers[first],
        bin_centers[second],
        bin_centers,
        sigma_cm,
        valid_bin_mask=valid_bin_mask,
    )
    n_bins = log_likelihood.shape[1]
    return (
        log_likelihood[0, first][:, None]
        + _uniform_log_prior(n_bins, valid_bin_mask)[first][:, None]
        + log_kernel
        + log_likelihood[1, second][None, :]
    )


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
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    coords_prev_prev = bin_centers[prev_prev]
    coords_prev = bin_centers[prev]
    coords_curr = bin_centers[curr]
    output = np.full((len(prev), len(curr)), LOG_ZERO, dtype=float)
    for prev_col in range(len(prev)):
        predictions = coords_prev[prev_col][None, :] + velocity_decay * (
            coords_prev[prev_col][None, :] - coords_prev_prev
        )
        log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
            predictions,
            coords_curr,
            bin_centers,
            sigma_cm,
            valid_bin_mask=valid_bin_mask,
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
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    coords_prev_prev = bin_centers[prev_prev]
    coords_prev = bin_centers[prev]
    coords_curr = bin_centers[curr]
    output = np.full((len(prev_prev), len(prev)), LOG_ZERO, dtype=float)
    for prev_col in range(len(prev)):
        predictions = coords_prev[prev_col][None, :] + velocity_decay * (
            coords_prev[prev_col][None, :] - coords_prev_prev
        )
        log_kernel = _full_grid_normalized_pairwise_gaussian_log_prob(
            predictions,
            coords_curr,
            bin_centers,
            sigma_cm,
            valid_bin_mask=valid_bin_mask,
        )
        continuation = curr_emission[None, :] + next_beta[prev_col][None, :]
        output[:, prev_col] = logsumexp(log_kernel + continuation, axis=1)
    return output
