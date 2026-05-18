"""Exact first-order state-space replay recursions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import _normalize_log_weights
from .state_space_utils import (
    _as_log_probs,
    _gaussian_transition_matrix,
    _mode_transition_matrix,
    _scaled_emissions,
)

TransitionSpec = csr_matrix | Sequence[csr_matrix]


def _score_stationary(emissions: LogEmissionTensor) -> tuple[float, np.ndarray]:
    log_weights = np.sum(emissions.log_likelihood, axis=0) - np.log(emissions.n_bins)
    logp = float(logsumexp(log_weights))
    posterior = _normalize_log_weights(log_weights)
    return logp, np.repeat(posterior[None, :], emissions.n_time, axis=0)


def _score_fragmented(emissions: LogEmissionTensor) -> tuple[float, np.ndarray]:
    scaled, offsets = _scaled_emissions(emissions.log_likelihood)
    row_sums = scaled.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("at least one emission row has no finite likelihood mass")
    logp = float(np.sum(np.log(row_sums / emissions.n_bins) + offsets))
    return logp, _as_log_probs(scaled / row_sums[:, None])


def _forward_backward_first_order(log_likelihood: np.ndarray, transition: TransitionSpec) -> tuple[float, np.ndarray]:
    n_time, n_bins = log_likelihood.shape
    scaled, offsets = _scaled_emissions(log_likelihood)
    filtered = np.zeros((n_time, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = scaled[0] / n_bins
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        step_transition = _transition_at(transition, time_index - 1)
        alpha = np.asarray(step_transition @ alpha, dtype=float) * scaled[time_index]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones(n_bins, dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        step_transition = _transition_at(transition, time_index - 1)
        beta = np.asarray(step_transition.T @ (scaled[time_index] * beta), dtype=float) / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]
    return logp, _as_log_probs(smoothed)


def _score_first_order_imm(
    log_likelihood: np.ndarray,
    bin_centers: np.ndarray,
    *,
    stationary_sigma_cm: float,
    diffusion_sigma_cm: float | np.ndarray,
    max_step_sigma: float,
    mode_stickiness: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    modes = ("stationary", "diffusion", "fragmented")
    n_modes = len(modes)
    n_time, n_bins = log_likelihood.shape
    transitions: dict[str, TransitionSpec | None] = {
        "stationary": _gaussian_transition_matrix(bin_centers, stationary_sigma_cm, max_step_sigma),
        "diffusion": _transition_spec_from_sigmas(bin_centers, diffusion_sigma_cm, max_step_sigma),
        "fragmented": None,
    }
    mode_transition = _mode_transition_matrix(n_modes, mode_stickiness)
    scaled, offsets = _scaled_emissions(log_likelihood)
    filtered = np.zeros((n_time, n_modes, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = np.tile(scaled[0] / (n_bins * n_modes), (n_modes, 1))
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        transition_index = time_index - 1
        predicted = np.zeros_like(alpha)
        for dst_idx, dst_mode in enumerate(modes):
            dst = np.zeros(n_bins, dtype=float)
            for src_idx in range(n_modes):
                dst += mode_transition[src_idx, dst_idx] * _apply_transition(
                    transitions[dst_mode],
                    alpha[src_idx],
                    transition_index=transition_index,
                )
            predicted[dst_idx] = dst
        alpha = predicted * scaled[time_index][None, :]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones((n_modes, n_bins), dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        transition_index = time_index - 1
        beta_prev = np.zeros_like(beta)
        for src_idx in range(n_modes):
            for dst_idx, dst_mode in enumerate(modes):
                beta_prev[src_idx] += mode_transition[src_idx, dst_idx] * _apply_transition_backward(
                    transitions[dst_mode],
                    scaled[time_index] * beta[dst_idx],
                    transition_index=transition_index,
                )
        beta = beta_prev / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]

    return logp, _as_log_probs(smoothed.sum(axis=1)), smoothed.sum(axis=2)


def _apply_transition(
    transition: TransitionSpec | None,
    weights: np.ndarray,
    *,
    transition_index: int = 0,
) -> np.ndarray:
    if transition is None:
        return np.full(weights.shape, float(weights.sum()) / weights.shape[0], dtype=float)
    return np.asarray(_transition_at(transition, transition_index) @ weights, dtype=float)


def _apply_transition_backward(
    transition: TransitionSpec | None,
    values: np.ndarray,
    *,
    transition_index: int = 0,
) -> np.ndarray:
    if transition is None:
        return np.full(values.shape, float(values.sum()) / values.shape[0], dtype=float)
    return np.asarray(_transition_at(transition, transition_index).T @ values, dtype=float)


def _transition_at(transition: TransitionSpec, transition_index: int) -> csr_matrix:
    if isinstance(transition, (list, tuple)):
        return transition[transition_index]
    return transition


def _transition_spec_from_sigmas(
    bin_centers: np.ndarray,
    sigmas_cm: float | np.ndarray,
    max_step_sigma: float,
) -> TransitionSpec:
    sigmas = np.asarray(sigmas_cm, dtype=float)
    if sigmas.ndim == 0:
        return _gaussian_transition_matrix(bin_centers, float(sigmas), max_step_sigma)
    if sigmas.ndim != 1:
        raise ValueError("diffusion_sigma_cm must be a scalar or one value per transition")
    return [_gaussian_transition_matrix(bin_centers, float(sigma), max_step_sigma) for sigma in sigmas]
