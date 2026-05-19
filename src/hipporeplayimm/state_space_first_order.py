"""Exact first-order state-space replay recursions."""

from __future__ import annotations

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

def _transition_at(transition: csr_matrix | list[csr_matrix] | tuple[csr_matrix, ...] | None, index: int):
    """Return the transition matrix for one center-to-center step.

    A scalar transition keeps the legacy fixed-bin behavior.  A sequence of
    transitions enables first-class variable-duration dynamics for ripples whose
    final bin is partial or whose bin centers are otherwise irregular.
    """
    if transition is None or not isinstance(transition, (list, tuple)):
        return transition
    return transition[index]

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


def _forward_backward_first_order(log_likelihood: np.ndarray, transition: csr_matrix) -> tuple[float, np.ndarray]:
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
    diffusion_sigma_cm: float,
    max_step_sigma: float,
    mode_stickiness: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    modes = ("stationary", "diffusion", "fragmented")
    n_modes = len(modes)
    n_time, n_bins = log_likelihood.shape
    diffusion_sigmas = np.asarray(diffusion_sigma_cm, dtype=float)
    if diffusion_sigmas.ndim == 0:
        diffusion_transition = _gaussian_transition_matrix(bin_centers, float(diffusion_sigmas), max_step_sigma)
    else:
        if diffusion_sigmas.shape != (max(n_time - 1, 0),):
            raise ValueError("diffusion_sigma_cm must be scalar or one value per transition")
        diffusion_transition = [
            _gaussian_transition_matrix(bin_centers, float(sigma), max_step_sigma) for sigma in diffusion_sigmas
        ]
    transitions = {
        "stationary": _gaussian_transition_matrix(bin_centers, stationary_sigma_cm, max_step_sigma),
        "diffusion": diffusion_transition,
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
        predicted = np.zeros_like(alpha)
        for dst_idx, dst_mode in enumerate(modes):
            dst = np.zeros(n_bins, dtype=float)
            for src_idx in range(n_modes):
                step_transition = _transition_at(transitions[dst_mode], time_index - 1)
                dst += mode_transition[src_idx, dst_idx] * _apply_transition(step_transition, alpha[src_idx])
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
        beta_prev = np.zeros_like(beta)
        for src_idx in range(n_modes):
            for dst_idx, dst_mode in enumerate(modes):
                step_transition = _transition_at(transitions[dst_mode], time_index - 1)
                beta_prev[src_idx] += mode_transition[src_idx, dst_idx] * _apply_transition_backward(
                    step_transition, scaled[time_index] * beta[dst_idx]
                )
        beta = beta_prev / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]

    return logp, _as_log_probs(smoothed.sum(axis=1)), smoothed.sum(axis=2)


def _apply_transition(transition: csr_matrix | None, weights: np.ndarray) -> np.ndarray:
    if transition is None:
        return np.full(weights.shape, float(weights.sum()) / weights.shape[0], dtype=float)
    return np.asarray(transition @ weights, dtype=float)


def _apply_transition_backward(transition: csr_matrix | None, values: np.ndarray) -> np.ndarray:
    if transition is None:
        return np.full(values.shape, float(values.sum()) / values.shape[0], dtype=float)
    return np.asarray(transition.T @ values, dtype=float)
