"""Exact sparse pair-grid momentum state-space decoder.

The existing candidate-pruned momentum decoder is exact only when every replay
time bin keeps the full spatial grid.  This module implements a separate
second-order dynamic program over pair states ``(x[t-1], x[t])`` with sparse,
finite-radius Gaussian transitions.  The evidence is exact for the declared
finite-radius transition model and is therefore comparable to the existing
exact first-order state-space baselines.

The implementation deliberately avoids emission top-k candidate support.  Sparse
support comes only from the transition model: for each source pair, destinations
within ``max_step_sigma * sigma`` of the momentum prediction are enumerated and
renormalized over the valid spatial grid.  If the finite radius contains no
valid spatial bin, the nearest valid bin is retained, matching the fallback used
by the first-order sparse Gaussian transition helper.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from .encoding import LogEmissionTensor
from .state_space_first_order import _score_fragmented
from .state_space_utils import _as_log_probs, _mean_entropy, _scaled_emissions

TransitionRows = list[tuple[np.ndarray, np.ndarray]]
TransitionRowCache = dict[tuple[int, int, float, float], tuple[np.ndarray, np.ndarray]]


def _score_sparse_momentum_exact(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    config: object,
    transition_durations_s: Iterable[float],
    *,
    valid_bin_mask: np.ndarray | None = None,
    return_trajectory: bool = True,
) -> tuple[float, np.ndarray | None, np.ndarray, dict[str, float | int | str]]:
    """Return exact evidence and position posteriors for sparse pair momentum.

    The latent state after the first transition is a sparse list of pair states
    ``(previous_position, current_position)``.  Forward and backward passes use
    the same finite-radius Gaussian transition rows, so the returned evidence is
    an exact HMM likelihood for the declared sparse transition model rather than
    a lower bound induced by decoder-likelihood candidate support.
    """

    centers = _as_2d_centers(bin_centers)
    if emissions.n_time <= 0:
        raise ValueError("emissions must contain at least one time bin")
    if emissions.n_bins != centers.shape[0]:
        raise ValueError("emissions.n_bins must match bin_centers rows")

    if emissions.n_time == 1:
        logp, trajectory = _score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
        diagnostics = _single_bin_diagnostics(config)
        if not return_trajectory:
            diagnostics["state_space_sparse_momentum_backward_transition_rows"] = "skipped_evidence_only"
            diagnostics["state_space_momentum_trajectory_posterior"] = "not_returned_evidence_only"
            return float(logp), None, trajectory[-1], diagnostics
        return float(logp), trajectory, trajectory[-1], diagnostics

    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, emissions.n_bins)
    valid_indices = _valid_indices(valid_mask, emissions.n_bins)
    log_likelihood = np.asarray(emissions.log_likelihood, dtype=float).copy()
    if valid_mask is not None:
        log_likelihood[:, ~valid_mask] = -np.inf

    durations = _coerce_transition_durations(
        transition_durations_s,
        n_time=emissions.n_time,
        fallback_dt=float(emissions.dt),
    )
    reference_dt = float(np.median(durations)) if durations.size else float(emissions.dt)
    transition_sigmas = _per_transition_sigmas(
        float(getattr(config, "momentum_sigma_cm_sqrt_s", 85.0)),
        durations,
    )
    initial_sigma = _per_bin_sigma(
        float(getattr(config, "momentum_initial_sigma_cm_sqrt_s", 85.0)),
        float(durations[0]) if durations.size else float(emissions.dt),
    )
    decays = _duration_adjusted_decays(config, durations, float(emissions.dt))
    time_scales = _time_scales(durations)
    max_step_sigma = float(getattr(config, "max_step_sigma", 4.0))
    if not np.isfinite(max_step_sigma) or max_step_sigma <= 0.0:
        raise ValueError("max_step_sigma must be finite and positive")

    tree = cKDTree(centers[valid_indices])
    prior = _uniform_position_prior(emissions.n_bins, valid_mask)

    def initial_pair_initializer(scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        return _initial_pair_alpha(
            centers,
            valid_indices,
            tree,
            scaled,
            prior,
            initial_sigma_cm=initial_sigma,
            max_step_sigma=max_step_sigma,
        )

    def transition_row_builder(src_prev: int, src_curr: int, transition_index: int) -> tuple[np.ndarray, np.ndarray]:
        prediction = centers[int(src_curr)] + float(decays[transition_index]) * float(time_scales[transition_index]) * (
            centers[int(src_curr)] - centers[int(src_prev)]
        )
        return _finite_gaussian_row(
            centers,
            valid_indices,
            tree,
            prediction,
            sigma_cm=float(transition_sigmas[transition_index]),
            max_step_sigma=max_step_sigma,
        )

    def transition_cache_key_builder(src_prev: int, src_curr: int, transition_index: int) -> tuple[int, int, float, float]:
        sigma = float(transition_sigmas[transition_index])
        velocity_decay = float(decays[transition_index]) * float(time_scales[transition_index])
        return int(src_prev), int(src_curr), sigma, velocity_decay

    del transition_row_builder, transition_cache_key_builder

    scaled, offsets = _scaled_emissions(log_likelihood)
    prev, curr, alpha, initial_edge_counts = initial_pair_initializer(scaled)
    scale = float(alpha.sum())
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("initial sparse momentum pair lattice has no finite mass")
    alpha = alpha / scale

    pair_prev: list[np.ndarray] = [prev]
    pair_curr: list[np.ndarray] = [curr]
    filtered: list[np.ndarray] = [alpha]
    scales: list[float] = [scale]
    transition_rows: list[TransitionRows] = []
    edge_counts: list[int] = list(initial_edge_counts)
    row_cache: TransitionRowCache = {}
    cache_hits = 0
    cache_misses = 0
    logp = float(np.log(scale) + offsets[0] + offsets[1])

    for time_index in range(2, emissions.n_time):
        transition_index = time_index - 1
        prev, curr, alpha, curr_edge_counts, rows, hits, misses = _advance_pair_alpha(
            centers,
            valid_indices,
            tree,
            prev,
            curr,
            alpha,
            scaled[time_index],
            sigma_cm=float(transition_sigmas[transition_index]),
            velocity_decay=float(decays[transition_index]) * float(time_scales[transition_index]),
            max_step_sigma=max_step_sigma,
            row_cache=row_cache,
            store_transition_rows=return_trajectory,
        )
        scale = float(alpha.sum())
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError(f"emission row {time_index} has no finite predicted sparse momentum mass")
        alpha = alpha / scale
        pair_prev.append(prev)
        pair_curr.append(curr)
        filtered.append(alpha)
        scales.append(scale)
        if return_trajectory:
            transition_rows.append(rows)
        edge_counts.extend(curr_edge_counts)
        cache_hits += hits
        cache_misses += misses
        logp += float(np.log(scale) + offsets[time_index])

    if return_trajectory:
        betas = _backward_sparse_pair_betas(
            centers,
            valid_indices,
            tree,
            pair_prev,
            pair_curr,
            filtered,
            scaled,
            scales,
            transition_rows,
        )
        trajectory = _pair_position_trajectory(
            pair_prev,
            pair_curr,
            filtered,
            betas,
            n_time=emissions.n_time,
            n_bins=emissions.n_bins,
        )
        terminal = trajectory[-1]
        trajectory_label = "smoothed_pair_marginal"
        posterior_entropy = _mean_entropy(trajectory)
    else:
        trajectory = None
        terminal = _terminal_position_log_posterior(pair_curr[-1], filtered[-1], n_bins=emissions.n_bins)
        trajectory_label = "not_returned_evidence_only"
        posterior_entropy = float("nan")
    pair_counts = np.asarray([values.shape[0] for values in filtered], dtype=float)
    outgoing_counts = np.asarray(edge_counts, dtype=float)
    backward_label = "computed_full_smoothing" if return_trajectory else "skipped_evidence_only"
    diagnostics: dict[str, float | int | str] = {
        "state_space_sparse_momentum_evidence_support": "exact_full_grid",
        "state_space_sparse_momentum_state_support": "finite_radius_pair_grid",
        "state_space_sparse_momentum_transition_support": "finite_radius_gaussian",
        "state_space_sparse_momentum_initial_pair_count": int(pair_counts[0]),
        "state_space_sparse_momentum_terminal_pair_count": int(pair_counts[-1]),
        "state_space_sparse_momentum_mean_pair_count": float(np.mean(pair_counts)),
        "state_space_sparse_momentum_max_pair_count": int(np.max(pair_counts)),
        "state_space_sparse_momentum_mean_outgoing_count": float(np.mean(outgoing_counts)) if outgoing_counts.size else 0.0,
        "state_space_sparse_momentum_max_outgoing_count": int(np.max(outgoing_counts)) if outgoing_counts.size else 0,
        "state_space_sparse_momentum_backward_transition_rows": backward_label,
        "state_space_sparse_momentum_evidence_mode": "full_smoothing" if return_trajectory else "evidence_only",
        "state_space_sparse_momentum_evidence_only": int(not return_trajectory),
        "state_space_sparse_momentum_transition_row_cache_entries": int(len(row_cache)),
        "state_space_sparse_momentum_transition_row_cache_hits": int(cache_hits),
        "state_space_sparse_momentum_transition_row_cache_misses": int(cache_misses),
        "state_space_momentum_trajectory_posterior": trajectory_label,
        "state_space_momentum_evidence_support": "exact_full_grid",
        "state_space_momentum_candidate_support": "not_used_exact_sparse",
        "state_space_momentum_candidate_selection": "none_exact_sparse",
        "state_space_momentum_transition_sigma_cm": _median_or_fallback(
            transition_sigmas,
            _per_bin_sigma(float(getattr(config, "momentum_sigma_cm_sqrt_s", 85.0)), reference_dt),
        ),
        "state_space_momentum_initial_transition_sigma_cm": float(initial_sigma),
        "state_space_momentum_transition_sigma_cm_per_step": _format_float_series(transition_sigmas),
        "state_space_momentum_velocity_decay_effective": _median_or_fallback(
            decays,
            float(getattr(config, "momentum_velocity_decay", 0.95)),
        ),
        "state_space_momentum_velocity_decay_per_step": _format_float_series(decays),
        "state_space_sparse_momentum_mean_posterior_entropy": posterior_entropy,
    }
    return logp, trajectory, terminal, diagnostics


def _single_bin_diagnostics(config: object) -> dict[str, float | int | str]:
    return {
        "state_space_sparse_momentum_evidence_support": "exact_full_grid",
        "state_space_sparse_momentum_state_support": "single_bin_fragmented_fallback",
        "state_space_sparse_momentum_transition_support": "none_single_bin",
        "state_space_sparse_momentum_initial_pair_count": 0,
        "state_space_sparse_momentum_terminal_pair_count": 0,
        "state_space_sparse_momentum_mean_pair_count": 0.0,
        "state_space_sparse_momentum_max_pair_count": 0,
        "state_space_sparse_momentum_mean_outgoing_count": 0.0,
        "state_space_sparse_momentum_max_outgoing_count": 0,
        "state_space_momentum_trajectory_posterior": "single_bin_fragmented_fallback",
        "state_space_momentum_evidence_support": "exact_full_grid",
        "state_space_momentum_candidate_support": "not_used_exact_sparse",
        "state_space_momentum_candidate_selection": "none_exact_sparse",
        "state_space_momentum_transition_sigma_cm": 0.0,
        "state_space_momentum_initial_transition_sigma_cm": 0.0,
        "state_space_momentum_velocity_decay_effective": float(getattr(config, "momentum_velocity_decay", 0.95)),
    }


def _initial_pair_alpha(
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    scaled: np.ndarray,
    prior: np.ndarray,
    *,
    initial_sigma_cm: float,
    max_step_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    prev_parts: list[np.ndarray] = []
    curr_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_counts: list[int] = []
    for src in valid_indices:
        if scaled[0, src] <= 0.0 or prior[src] <= 0.0:
            continue
        dst, weights = _finite_gaussian_row(
            centers,
            valid_indices,
            tree,
            centers[int(src)],
            sigma_cm=initial_sigma_cm,
            max_step_sigma=max_step_sigma,
        )
        values = prior[src] * scaled[0, src] * weights * scaled[1, dst]
        keep = values > 0.0
        edge_counts.append(int(dst.shape[0]))
        if not np.any(keep):
            continue
        prev_parts.append(np.full(int(np.sum(keep)), int(src), dtype=int))
        curr_parts.append(dst[keep])
        value_parts.append(values[keep])
    return (*_coalesce_pairs(prev_parts, curr_parts, value_parts, centers.shape[0]), edge_counts)


def _advance_pair_alpha(
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    prev: np.ndarray,
    curr: np.ndarray,
    alpha: np.ndarray,
    curr_emission: np.ndarray,
    *,
    sigma_cm: float,
    velocity_decay: float,
    max_step_sigma: float,
    row_cache: TransitionRowCache | None = None,
    store_transition_rows: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], TransitionRows, int, int]:
    prev_parts: list[np.ndarray] = []
    curr_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_counts: list[int] = []
    transition_rows: TransitionRows = []
    cache_hits = 0
    cache_misses = 0
    for src_prev, src_curr, source_mass in zip(prev, curr, alpha, strict=True):
        if source_mass <= 0.0:
            if store_transition_rows:
                transition_rows.append((np.empty(0, dtype=int), np.empty(0, dtype=float)))
            continue
        cache_key = (int(src_prev), int(src_curr), float(sigma_cm), float(velocity_decay))
        cached = None if row_cache is None else row_cache.get(cache_key)
        if cached is None:
            prediction = centers[int(src_curr)] + float(velocity_decay) * (centers[int(src_curr)] - centers[int(src_prev)])
            dst, weights = _finite_gaussian_row(
                centers,
                valid_indices,
                tree,
                prediction,
                sigma_cm=sigma_cm,
                max_step_sigma=max_step_sigma,
            )
            if row_cache is not None:
                row_cache[cache_key] = (dst, weights)
            cache_misses += 1
        else:
            dst, weights = cached
            cache_hits += 1
        if store_transition_rows:
            transition_rows.append((dst, weights))
        values = float(source_mass) * weights * curr_emission[dst]
        keep = values > 0.0
        edge_counts.append(int(dst.shape[0]))
        if not np.any(keep):
            continue
        prev_parts.append(np.full(int(np.sum(keep)), int(src_curr), dtype=int))
        curr_parts.append(dst[keep])
        value_parts.append(values[keep])
    return (
        *_coalesce_pairs(prev_parts, curr_parts, value_parts, centers.shape[0]),
        edge_counts,
        transition_rows,
        cache_hits,
        cache_misses,
    )


def _backward_sparse_pair_betas(
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    pair_prev: list[np.ndarray],
    pair_curr: list[np.ndarray],
    filtered: list[np.ndarray],
    scaled: np.ndarray,
    scales: list[float],
    transition_rows: list[TransitionRows],
) -> list[np.ndarray]:
    betas: list[np.ndarray] = [np.empty(0, dtype=float) for _ in filtered]
    betas[-1] = np.ones_like(filtered[-1], dtype=float)
    if len(transition_rows) != max(len(filtered) - 1, 0):
        raise ValueError("transition row cache length does not match sparse pair lattice")
    n_bins = centers.shape[0]
    for pair_index in range(len(filtered) - 2, -1, -1):
        next_flat = pair_prev[pair_index + 1].astype(np.int64) * n_bins + pair_curr[pair_index + 1].astype(np.int64)
        order = np.argsort(next_flat, kind="stable")
        next_flat = next_flat[order]
        next_beta = betas[pair_index + 1][order]
        beta = np.zeros_like(filtered[pair_index], dtype=float)
        observation_index = pair_index + 2
        rows_for_transition = transition_rows[pair_index]
        if len(rows_for_transition) != len(filtered[pair_index]):
            raise ValueError("transition row cache does not match sparse pair count")
        for row, (src_prev, src_curr) in enumerate(zip(pair_prev[pair_index], pair_curr[pair_index], strict=True)):
            dst, weights = rows_for_transition[row]
            query = int(src_curr) * n_bins + dst.astype(np.int64)
            continuation = _lookup_sorted(next_flat, next_beta, query)
            beta[row] = float(np.sum(weights * scaled[observation_index, dst] * continuation) / scales[pair_index + 1])
        betas[pair_index] = beta
    return betas


def _pair_position_trajectory(
    pair_prev: list[np.ndarray],
    pair_curr: list[np.ndarray],
    filtered: list[np.ndarray],
    betas: list[np.ndarray],
    *,
    n_time: int,
    n_bins: int,
) -> np.ndarray:
    position = np.zeros((n_time, n_bins), dtype=float)
    for pair_index, (prev, curr, alpha, beta) in enumerate(zip(pair_prev, pair_curr, filtered, betas, strict=True)):
        posterior = np.asarray(alpha, dtype=float) * np.asarray(beta, dtype=float)
        total = float(posterior.sum())
        if total > 0.0 and np.isfinite(total):
            posterior = posterior / total
        if pair_index == 0:
            np.add.at(position[0], prev, posterior)
        np.add.at(position[pair_index + 1], curr, posterior)
    for time_index in range(n_time):
        total = float(position[time_index].sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError(f"sparse momentum posterior at time {time_index} has no finite mass")
        position[time_index] /= total
    return _as_log_probs(position)


def _terminal_position_log_posterior(curr: np.ndarray, alpha: np.ndarray, *, n_bins: int) -> np.ndarray:
    position = np.zeros(int(n_bins), dtype=float)
    np.add.at(position, curr, alpha)
    total = float(position.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("sparse momentum terminal posterior has no finite mass")
    position /= total
    return _as_log_probs(position[None, :])[0]


def _finite_gaussian_row(
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    predicted: np.ndarray,
    *,
    sigma_cm: float,
    max_step_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(sigma_cm) or sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be finite and positive")
    predicted = np.asarray(predicted, dtype=float).reshape(centers.shape[1])
    radius = max(float(sigma_cm) * float(max_step_sigma), np.finfo(float).eps)
    local = tree.query_ball_point(predicted, radius)
    if len(local) == 0:
        _, nearest = tree.query(predicted, k=1)
        local = [int(nearest)]
    dst = valid_indices[np.asarray(local, dtype=int)]
    dist2 = np.sum((centers[dst] - predicted[None, :]) ** 2, axis=1)
    weights = np.exp(-0.5 * dist2 / max(float(sigma_cm) ** 2, np.finfo(float).tiny))
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        weights = np.ones(dst.shape[0], dtype=float) / max(int(dst.shape[0]), 1)
    else:
        weights /= total
    return dst.astype(int), weights


def _coalesce_pairs(
    prev_parts: list[np.ndarray],
    curr_parts: list[np.ndarray],
    value_parts: list[np.ndarray],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not value_parts:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)
    prev = np.concatenate(prev_parts).astype(np.int64, copy=False)
    curr = np.concatenate(curr_parts).astype(np.int64, copy=False)
    values = np.concatenate(value_parts).astype(float, copy=False)
    keep = np.isfinite(values) & (values > 0.0)
    if not np.any(keep):
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)
    flat = prev[keep] * int(n_bins) + curr[keep]
    values = values[keep]
    order = np.argsort(flat, kind="stable")
    flat = flat[order]
    values = values[order]
    unique, starts = np.unique(flat, return_index=True)
    summed = np.add.reduceat(values, starts)
    keep_summed = summed > 0.0
    unique = unique[keep_summed]
    summed = summed[keep_summed]
    return (unique // int(n_bins)).astype(int), (unique % int(n_bins)).astype(int), summed


def _lookup_sorted(sorted_keys: np.ndarray, values: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
    query = np.asarray(query_keys, dtype=np.int64)
    positions = np.searchsorted(sorted_keys, query)
    out = np.zeros(query.shape[0], dtype=float)
    in_bounds = positions < sorted_keys.shape[0]
    if not np.any(in_bounds):
        return out
    query_rows = np.flatnonzero(in_bounds)
    matched = sorted_keys[positions[query_rows]] == query[query_rows]
    out[query_rows[matched]] = values[positions[query_rows[matched]]]
    return out


def _uniform_position_prior(n_bins: int, valid_bin_mask: np.ndarray | None) -> np.ndarray:
    prior = np.zeros(n_bins, dtype=float)
    if valid_bin_mask is None:
        prior.fill(1.0 / n_bins)
    else:
        prior[valid_bin_mask] = 1.0 / int(np.sum(valid_bin_mask))
    return prior


def _as_2d_centers(bin_centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2:
        raise ValueError("bin_centers must be a one- or two-dimensional array")
    return centers


def _coerce_valid_bin_mask(mask: np.ndarray | None, n_bins: int) -> np.ndarray | None:
    if mask is None:
        return None
    out = np.asarray(mask, dtype=bool)
    if out.shape != (n_bins,):
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(out):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    return out


def _valid_indices(mask: np.ndarray | None, n_bins: int) -> np.ndarray:
    return np.arange(n_bins, dtype=int) if mask is None else np.flatnonzero(mask).astype(int)


def _coerce_transition_durations(values: Iterable[float], *, n_time: int, fallback_dt: float) -> np.ndarray:
    expected = max(int(n_time) - 1, 0)
    out = np.asarray(list(values), dtype=float)
    if out.shape != (expected,) or not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        return np.full(expected, float(fallback_dt), dtype=float)
    return out


def _per_transition_sigmas(sigma_cm_sqrt_s: float, durations: np.ndarray) -> np.ndarray:
    return np.asarray([_per_bin_sigma(sigma_cm_sqrt_s, duration) for duration in durations], dtype=float)


def _per_bin_sigma(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    sigma = float(sigma_cm_sqrt_s)
    dt = float(dt_s)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_cm_sqrt_s must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    return max(sigma * np.sqrt(dt), np.finfo(float).eps)


def _duration_adjusted_decays(config: object, durations: np.ndarray, reference_dt: float) -> np.ndarray:
    if durations.size == 0:
        return np.empty(0, dtype=float)
    tau_s = float(getattr(config, "momentum_velocity_decay_tau_s", 0.0))
    if tau_s > 0.0:
        if not np.isfinite(tau_s):
            raise ValueError("momentum_velocity_decay_tau_s must be finite when positive")
        return np.exp(durations * (-1.0 / tau_s))
    decay = float(getattr(config, "momentum_velocity_decay", 0.95))
    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError("momentum_velocity_decay must be finite and nonnegative")
    if not np.isfinite(reference_dt) or reference_dt <= 0.0:
        raise ValueError("reference dt must be finite and positive")
    return np.asarray([decay ** (float(duration) / reference_dt) for duration in durations], dtype=float)


def _time_scales(durations: np.ndarray) -> np.ndarray:
    scales = np.ones_like(durations, dtype=float)
    if durations.size > 1:
        scales[1:] = durations[1:] / durations[:-1]
    return scales


def _median_or_fallback(values: np.ndarray, fallback: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(fallback) if arr.size == 0 else float(np.median(arr))


def _format_float_series(values: np.ndarray) -> str:
    return ",".join(f"{float(value):.12g}" for value in np.asarray(values, dtype=float))
