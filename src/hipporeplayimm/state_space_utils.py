"""Shared helpers for state-space replay decoders."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

LOG_ZERO = -1.0e300


def _is_boolean_scalar(value: object) -> bool:
    """Return True for Python, NumPy, and object-wrapped boolean scalars."""

    if isinstance(value, (bool, np.bool_)):
        return True
    arr = np.asarray(value)
    if arr.ndim != 0:
        return False
    if np.issubdtype(arr.dtype, np.bool_):
        return True
    if arr.dtype == object:
        try:
            return isinstance(arr.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _reject_boolean_count(name: str, value: object) -> None:
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be an integer count, not boolean")


def _per_bin_sigma(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    sigma = float(sigma_cm_sqrt_s)
    dt = float(dt_s)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_cm_sqrt_s must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    return max(sigma * np.sqrt(dt), np.finfo(float).eps)


def _top_candidate_indices(log_emission: np.ndarray, top_k: int) -> np.ndarray:
    _reject_boolean_count("top_k", top_k)
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

    if top_k is not None:
        _reject_boolean_count("top_k", top_k)
    _reject_boolean_count("min_k", min_k)
    _reject_boolean_count("max_k", max_k)

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
    order = np.argsort(np.where(finite, values, -np.inf))[::-1]
    finite_order = order[finite[order]]
    finite_count = int(finite_order.size)
    top_k_minimum = 0 if top_k is None or int(top_k) <= 0 else int(top_k)
    min_count = min(finite_count, max(1, top_k_minimum, int(min_k)))
    max_count = (
        finite_count if max_k <= 0 else min(finite_count, max(min_count, int(max_k)))
    )
    ordered_values = values[finite_order]
    cumulative_mass = np.cumsum(np.exp(ordered_values - logsumexp(ordered_values)))
    tolerance = 16.0 * np.finfo(float).eps
    mass_count = int(
        np.searchsorted(cumulative_mass + tolerance, float(mass_threshold), side="left") + 1
    )
    count = min(max(min_count, mass_count), max_count)
    return np.asarray(finite_order[:count], dtype=int)


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
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("min_occupancy_s must be finite and nonnegative")
    if threshold == 0.0 or occupancy_s is None:
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
    n_time, n_bins = log_likelihood.shape
    validated = _validate_candidate_indices(candidates, n_time, n_bins)
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_bins)
    if valid_mask is None:
        return validated
    valid_indices = np.flatnonzero(valid_mask)
    restricted: list[np.ndarray] = []
    for time_index, arr in enumerate(validated):
        keep = arr[valid_mask[arr]]
        if keep.size == 0:
            valid_scores = log_likelihood[time_index, valid_indices]
            keep = np.asarray([valid_indices[int(np.argmax(valid_scores))]], dtype=int)
        restricted.append(np.unique(keep.astype(int)))
    return restricted


def _as_finite_2d_points(values: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_points, position_dim)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _gaussian_transition_matrix(
    bin_centers: np.ndarray,
    sigma_cm: float,
    max_step_sigma: float,
    valid_bin_mask: np.ndarray | None = None,
) -> csr_matrix:
    sigma_cm = float(sigma_cm)
    max_step_sigma = float(max_step_sigma)
    if not np.isfinite(sigma_cm) or sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be finite and positive")
    if not np.isfinite(max_step_sigma) or max_step_sigma <= 0.0:
        raise ValueError("max_step_sigma must be finite and positive")
    bin_centers = _as_finite_2d_points(bin_centers, "bin_centers")
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
        weights_sum = float(weights.sum())
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            weights = np.ones(dst.shape[0], dtype=float) / max(int(dst.shape[0]), 1)
        else:
            weights /= weights_sum
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

    all_observed = _as_finite_2d_points(all_observed, "all_observed")
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
    sigma_cm = float(sigma_cm)
    if not np.isfinite(sigma_cm) or sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be finite and positive")
    predicted = _as_finite_2d_points(predicted, "predicted")
    observed = _as_finite_2d_points(observed, "observed")
    if predicted.shape[1] != observed.shape[1]:
        raise ValueError("predicted and observed must have matching position dimensions")
    delta = predicted[:, None, :] - observed[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    return -0.5 * dist2 / (sigma_cm * sigma_cm)


def _scaled_emissions(
    log_likelihood: np.ndarray,
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("log_likelihood must be two-dimensional")
    if np.any(np.isnan(values)) or np.any(values == np.inf):
        raise ValueError("log_likelihood must not contain NaN or +inf")

    finite = np.isfinite(values)
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, values.shape[1])
    if valid_mask is not None:
        finite &= valid_mask[None, :]

    if not np.all(np.any(finite, axis=1)):
        raise ValueError(
            "every emission row must contain at least one finite value on the active support"
        )

    offsets = np.max(np.where(finite, values, -np.inf), axis=1)
    shifted = np.where(finite, values - offsets[:, None], -np.inf)
    scaled = np.exp(np.clip(shifted, -745.0, 0.0))
    scaled[~finite] = 0.0
    return scaled, offsets


def _as_log_probs(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be nonnegative")

    row_mass = probs.sum(axis=1)
    if not np.all(np.isfinite(row_mass)) or np.any(row_mass <= 0.0):
        raise ValueError("every probability row must contain positive finite mass")

    out = np.full(probs.shape, LOG_ZERO, dtype=float)
    normalized = probs / row_mass[:, None]
    positive = normalized > 0.0
    out[positive] = np.log(normalized[positive])
    return out


def _mean_entropy(trajectory_log_posterior: np.ndarray) -> float:
    log_posterior = np.asarray(trajectory_log_posterior, dtype=float)
    posterior = np.exp(log_posterior)
    with np.errstate(invalid="ignore"):
        entropy_terms = np.where(posterior > 0.0, posterior * log_posterior, 0.0)
    return float(np.mean(-np.sum(entropy_terms, axis=1)))


def _diagnostic_transition_durations(
    dt_s: float,
    n_time: int,
    fallback_dt_s: float,
) -> np.ndarray:
    n_transitions = max(int(n_time) - 1, 0)
    transition_durations = getattr(dt_s, "transition_durations", None)
    if transition_durations is None:
        return np.full(n_transitions, float(fallback_dt_s), dtype=float)

    durations = np.asarray(transition_durations, dtype=float)
    if durations.shape != (n_transitions,):
        raise ValueError("transition_durations must contain one value per adjacent time-bin pair")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("transition_durations must contain finite positive durations")
    return durations


def _longest_active_run_duration(
    active: np.ndarray,
    transition_durations: np.ndarray,
    fallback_dt_s: float,
) -> float:
    active = np.asarray(active, dtype=bool)
    if active.size == 0:
        return 0.0
    bin_duration = (
        float(np.median(transition_durations)) if transition_durations.size else float(fallback_dt_s)
    )
    best = 0.0
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is None:
            continue
        if value and index != active.size - 1:
            continue
        stop = index if value else index - 1
        duration = bin_duration
        if stop > start:
            duration += float(np.sum(transition_durations[start:stop]))
        best = max(best, duration)
        start = None
    return best


def _first_order_imm_content_diagnostics(
    mode_posterior: np.ndarray,
    trajectory_log_posterior: np.ndarray,
    bin_centers: np.ndarray,
    dt_s: float,
) -> dict[str, float | int]:
    mode = np.asarray(mode_posterior, dtype=float)
    trajectory = np.asarray(trajectory_log_posterior, dtype=float)
    centers = np.asarray(bin_centers, dtype=float)
    if mode.ndim != 2 or mode.shape[1] != 3:
        raise ValueError("first-order IMM mode posterior must have shape (time, 3)")
    if trajectory.ndim != 2 or trajectory.shape[0] != mode.shape[0]:
        raise ValueError("trajectory posterior must have one row per mode-posterior time bin")
    if centers.ndim != 2 or centers.shape[0] != trajectory.shape[1] or centers.shape[1] < 1:
        raise ValueError("bin_centers must contain one coordinate row per spatial bin")
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    transition_durations = _diagnostic_transition_durations(dt_s, mode.shape[0], dt)

    map_mode = np.argmax(mode, axis=1)
    nonstationary = map_mode != 0
    starts = nonstationary & np.concatenate(([True], ~nonstationary[:-1]))
    bout_count = int(starts.sum())
    longest_duration = _longest_active_run_duration(nonstationary, transition_durations, dt)

    posterior = np.exp(trajectory)
    row_mass = posterior.sum(axis=1)
    valid = row_mass > 0.0
    posterior[valid] = posterior[valid] / row_mass[valid, None]
    expected_position = posterior @ centers
    if len(expected_position) > 1:
        steps = np.linalg.norm(np.diff(expected_position, axis=0), axis=1)
        path_length = float(np.nansum(steps))
        net = float(np.linalg.norm(expected_position[-1] - expected_position[0]))
        duration = max(float(np.sum(transition_durations)), np.finfo(float).tiny)
    else:
        path_length = 0.0
        net = 0.0
        duration = dt

    return {
        "state_space_imm_fraction_time_map_stationary": float(np.mean(~nonstationary)),
        "state_space_imm_fraction_time_map_nonstationary": float(np.mean(nonstationary)),
        "state_space_imm_nonstationary_bout_count": bout_count,
        "state_space_imm_longest_nonstationary_bout_s": longest_duration,
        "state_space_imm_posterior_expected_path_length_cm": path_length,
        "state_space_imm_posterior_net_displacement_cm": net,
        "state_space_imm_posterior_path_speed_cm_s": path_length / duration,
    }


def _mode_transition_matrix(n_modes: int, stickiness: float) -> np.ndarray:
    n_modes = int(n_modes)
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    if not 0.0 <= stickiness <= 1.0:
        raise ValueError("mode_stickiness must be in [0, 1]")
    if n_modes == 1:
        return np.ones((1, 1), dtype=float)
    off_diag = (1.0 - stickiness) / (n_modes - 1)
    matrix = np.full((n_modes, n_modes), off_diag, dtype=float)
    np.fill_diagonal(matrix, stickiness)
    return matrix
