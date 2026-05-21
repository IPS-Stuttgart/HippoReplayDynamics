"""Shared helpers for state-space replay decoders."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

LOG_ZERO = -1.0e300


def _per_bin_sigma(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    sigma = float(sigma_cm_sqrt_s)
    dt = float(dt_s)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_cm_sqrt_s must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    return max(sigma * np.sqrt(dt), np.finfo(float).eps)


def _top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= log_emission.shape[0]:
        return np.arange(log_emission.shape[0], dtype=int)
    selected = np.argpartition(log_emission, -top_k)[-top_k:]
    return selected[np.argsort(log_emission[selected])[::-1]]


def _mass_retaining_candidate_indices(
    log_emission: np.ndarray,
    mass_threshold: float | None = None,
    *,
    top_k: int | None = None,
    min_k: int = 1,
    max_k: int = 0,
) -> np.ndarray:
    """Return emission candidates that retain a target normalized mass.

    ``top_k`` is an optional legacy lower bound; when ``mass_threshold`` is
    disabled it is used as fixed top-k support. ``max_k <= 0`` means unbounded.
    The returned indices are sorted by decreasing emission log-likelihood,
    matching ``_top_candidate_indices``.
    """

    values = np.asarray(log_emission, dtype=float)
    if values.ndim != 1:
        raise ValueError("log_emission must be one-dimensional")
    if values.size == 0:
        return np.empty(0, dtype=int)
    if mass_threshold is None or float(mass_threshold) <= 0.0:
        return _top_candidate_indices(values, 0 if top_k is None else int(top_k))
    if not 0.0 < float(mass_threshold) <= 1.0:
        raise ValueError("mass_threshold must be in (0, 1]")
    if min_k < 0:
        raise ValueError("min_k must be non-negative")
    if max_k < 0:
        raise ValueError("max_k must be non-negative")
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("log_emission must contain at least one finite value")
    n_bins = values.shape[0]
    top_k_minimum = 0 if top_k is None or int(top_k) <= 0 else int(top_k)
    min_count = min(n_bins, max(1, top_k_minimum, int(min_k)))
    max_count = (
        n_bins if max_k <= 0 else min(n_bins, max(min_count, int(max_k)))
    )
    order = np.argsort(np.where(finite, values, -np.inf))[::-1]
    ordered_values = np.where(finite[order], values[order], -np.inf)
    cumulative_mass = np.cumsum(np.exp(ordered_values - logsumexp(ordered_values)))
    tolerance = 16.0 * np.finfo(float).eps
    mass_count = int(
        np.searchsorted(cumulative_mass + tolerance, float(mass_threshold), side="left") + 1
    )
    count = min(max(min_count, mass_count), max_count)
    return np.asarray(order[:count], dtype=int)


def _validate_candidate_indices(
    candidates: list[np.ndarray],
    n_time: int,
    n_bins: int,
) -> list[np.ndarray]:
    """Validate and canonicalize candidate supports as integer index arrays."""

    if len(candidates) != n_time:
        raise ValueError("candidate_indices must contain one array per emission time bin")
    validated: list[np.ndarray] = []
    for time_index, curr in enumerate(candidates):
        arr = np.asarray(curr)
        if arr.ndim != 1:
            raise ValueError(f"candidate_indices[{time_index}] must be one-dimensional")
        if arr.size == 0:
            raise ValueError(f"candidate_indices[{time_index}] must not be empty")
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"candidate_indices[{time_index}] must contain integer bin indices")
        arr = arr.astype(np.intp, copy=False)
        if np.any((arr < 0) | (arr >= n_bins)):
            raise ValueError(f"candidate_indices[{time_index}] contains an out-of-range bin")
        if np.unique(arr).size != arr.size:
            raise ValueError(f"candidate_indices[{time_index}] contains duplicate bins")
        validated.append(arr)
    return validated


def _candidate_log_masses(log_likelihood: np.ndarray, candidates: list[np.ndarray]) -> list[float]:
    return [
        float(logsumexp(log_likelihood[time_index, curr]) - logsumexp(log_likelihood[time_index]))
        for time_index, curr in enumerate(candidates)
    ]


def _coerce_valid_bin_mask(valid_bin_mask: np.ndarray | None, n_bins: int) -> np.ndarray | None:
    if valid_bin_mask is None:
        return None
    mask = np.asarray(valid_bin_mask, dtype=bool)
    if mask.shape != (n_bins,):
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(mask):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    return mask


def _valid_bin_mask_from_occupancy(
    occupancy_s: np.ndarray | None,
    min_occupancy_s: float,
    n_bins: int,
) -> np.ndarray | None:
    threshold = float(min_occupancy_s)
    if threshold <= 0.0 or occupancy_s is None:
        return None
    occupancy = np.asarray(occupancy_s, dtype=float)
    if occupancy.shape != (n_bins,):
        raise ValueError("occupancy_s must contain one value per spatial bin")
    mask = np.isfinite(occupancy) & (occupancy >= threshold)
    if not np.any(mask):
        raise ValueError("occupancy threshold excludes every spatial bin")
    return mask


def _uniform_log_prior(n_bins: int, valid_bin_mask: np.ndarray | None = None) -> np.ndarray:
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_bins)
    log_prior = np.full(n_bins, LOG_ZERO, dtype=float)
    if valid_mask is None:
        log_prior.fill(-np.log(n_bins))
    else:
        log_prior[valid_mask] = -np.log(int(np.sum(valid_mask)))
    return log_prior


def _uniform_probabilities(n_bins: int, valid_bin_mask: np.ndarray | None = None) -> np.ndarray:
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_bins)
    probs = np.zeros(n_bins, dtype=float)
    if valid_mask is None:
        probs.fill(1.0 / n_bins)
    else:
        probs[valid_mask] = 1.0 / int(np.sum(valid_mask))
    return probs


def _valid_bin_count(n_bins: int, valid_bin_mask: np.ndarray | None = None) -> int:
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_bins)
    return n_bins if valid_mask is None else int(np.sum(valid_mask))


def _restrict_candidates_to_valid_bins(
    candidates: list[np.ndarray],
    log_likelihood: np.ndarray,
    valid_bin_mask: np.ndarray | None,
) -> list[np.ndarray]:
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, log_likelihood.shape[1])
    if valid_mask is None:
        return candidates
    valid_indices = np.flatnonzero(valid_mask)
    restricted: list[np.ndarray] = []
    for time_index, curr in enumerate(candidates):
        arr = np.asarray(curr, dtype=int)
        keep = arr[valid_mask[arr]]
        if keep.size == 0:
            valid_scores = log_likelihood[time_index, valid_indices]
            keep = np.asarray([valid_indices[int(np.argmax(valid_scores))]], dtype=int)
        restricted.append(np.unique(keep.astype(int)))
    return restricted


def _gaussian_transition_matrix(
    bin_centers: np.ndarray,
    sigma_cm: float,
    max_step_sigma: float,
    valid_bin_mask: np.ndarray | None = None,
) -> csr_matrix:
    if sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be positive")
    n_bins = bin_centers.shape[0]
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_bins)
    allowed = np.arange(n_bins, dtype=int) if valid_mask is None else np.flatnonzero(valid_mask)
    radius2 = (sigma_cm * max_step_sigma) ** 2
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src, center in enumerate(bin_centers):
        delta = bin_centers - center[None, :]
        dist2 = np.sum(delta * delta, axis=1)
        keep = dist2 <= radius2
        if valid_mask is not None:
            keep &= valid_mask
        if not np.any(keep):
            keep[int(allowed[int(np.argmin(dist2[allowed]))])] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * dist2[dst] / (sigma_cm * sigma_cm))
        weights /= float(weights.sum())
        rows.extend(int(idx) for idx in dst)
        cols.extend([src] * len(dst))
        data.extend(float(value) for value in weights)
    return csr_matrix((data, (rows, cols)), shape=(n_bins, n_bins))


def _full_grid_normalized_pairwise_gaussian_log_prob(
    predicted: np.ndarray,
    observed: np.ndarray,
    all_observed: np.ndarray,
    sigma_cm: float,
    valid_bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate candidate log transitions with full-grid normalization."""

    log_kernel = _pairwise_gaussian_log_prob(predicted, observed, sigma_cm)
    normalizer_support = all_observed
    if valid_bin_mask is not None:
        valid_mask = _coerce_valid_bin_mask(valid_bin_mask, all_observed.shape[0])
        assert valid_mask is not None
        normalizer_support = all_observed[valid_mask]
    log_normalizer = logsumexp(
        _pairwise_gaussian_log_prob(predicted, normalizer_support, sigma_cm),
        axis=1,
        keepdims=True,
    )
    return log_kernel - log_normalizer


def _pairwise_gaussian_log_prob(predicted: np.ndarray, observed: np.ndarray, sigma_cm: float) -> np.ndarray:
    delta = predicted[:, None, :] - observed[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    return -0.5 * dist2 / (sigma_cm * sigma_cm)


def _scaled_emissions(log_likelihood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(log_likelihood, dtype=float)
    finite = np.isfinite(values)
    if not np.all(np.any(finite, axis=1)):
        raise ValueError("every emission row must contain at least one finite value")
    offsets = np.max(np.where(finite, values, -np.inf), axis=1)
    shifted = np.where(finite, values - offsets[:, None], -np.inf)
    scaled = np.exp(np.clip(shifted, -745.0, 0.0))
    scaled[~finite] = 0.0
    return scaled, offsets


def _as_log_probs(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    out = np.full(probs.shape, LOG_ZERO, dtype=float)
    positive = probs > 0.0
    out[positive] = np.log(probs[positive])
    return out - logsumexp(out, axis=1)[:, None]


def _mean_entropy(trajectory_log_posterior: np.ndarray) -> float:
    posterior = np.exp(trajectory_log_posterior)
    return float(np.mean(-np.sum(posterior * trajectory_log_posterior, axis=1)))


def _mode_transition_matrix(n_modes: int, stickiness: float) -> np.ndarray:
    if n_modes < 2:
        return np.ones((n_modes, n_modes), dtype=float)
    if not 0.0 <= stickiness <= 1.0:
        raise ValueError("mode_stickiness must be in [0, 1]")
    off_diag = (1.0 - stickiness) / (n_modes - 1)
    matrix = np.full((n_modes, n_modes), off_diag, dtype=float)
    np.fill_diagonal(matrix, stickiness)
    return matrix
