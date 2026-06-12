"""Krause-Drugowitsch-style replay model-evidence scoring.

This module keeps the reference-aligned path separate from the repository's
candidate-pruned IMM benchmark. The implementation follows the public
HippocampalSWRDynamics model definitions closely enough to support alignment
experiments without vendoring that repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln, logsumexp
from scipy.stats import invgamma, multivariate_normal

from .data import ReplaySession, RippleEvent, _coerce_ripple_event
from .encoding import (
    EncodingModel,
    LogEmissionTensor,
    _clean_position,
    _frame_durations,
    _interp_positions,
    _positions_to_flat_bins,
    _speed_cm_s,
    _smooth_count_rows,
    _validate_poisson_inputs,
    _times_in_intervals,
)


KD_MODELS = ("random", "stationary", "stationary-gaussian", "diffusion", "momentum")
TRAJECTORY_MODELS = {"diffusion", "momentum"}
NONTRAJECTORY_MODELS = {"random", "stationary", "stationary-gaussian"}


@dataclass(frozen=True)
class KDEncodingConfig:
    bin_size_cm: float = 4.0
    n_bins_x: int = 50
    n_bins_y: int = 50
    smoothing_sigma_cm: float = 4.0
    min_speed_cm_s: float = 5.0
    min_occupancy_s: float = 0.02
    rate_floor_hz: float = 1e-4
    min_peak_rate_hz: float = 2.0
    use_excitatory: bool = True


@dataclass(frozen=True)
class KDGridConfig:
    diffusion_sd_meters: np.ndarray
    stationary_gaussian_sd_meters: np.ndarray
    momentum_sd_meters: np.ndarray
    momentum_decay: np.ndarray
    momentum_initial_sd_m_per_s: float = 10.0

    @classmethod
    def ripple_defaults(cls) -> "KDGridConfig":
        return cls(
            diffusion_sd_meters=np.logspace(-1, 0.8, 30).round(2),
            stationary_gaussian_sd_meters=np.logspace(-2, 0.3, 30).round(2),
            momentum_sd_meters=np.logspace(1.6, 2.6, 30).round(2),
            momentum_decay=np.array([1, 25, 50, 75, 100, 200, 300, 400, 500, 800], dtype=float),
        )

    @classmethod
    def smoke(cls) -> "KDGridConfig":
        return cls(
            diffusion_sd_meters=np.array([0.1, 0.5, 2.0], dtype=float),
            stationary_gaussian_sd_meters=np.array([0.01, 0.05, 0.2], dtype=float),
            momentum_sd_meters=np.array([40.0, 120.0], dtype=float),
            momentum_decay=np.array([1.0, 100.0], dtype=float),
        )


def grid_config_for_preset(preset: str) -> KDGridConfig:
    normalized = preset.strip().lower()
    if normalized == "kd":
        return KDGridConfig.ripple_defaults()
    if normalized == "smoke":
        return KDGridConfig.smoke()
    raise ValueError("grid preset must be one of: kd, smoke")


def fit_kd_place_field_encoding(session: ReplaySession, config: KDEncodingConfig | None = None) -> EncodingModel:
    config = KDEncodingConfig() if config is None else config
    position = _clean_position(session.position)
    times = position[:, 0]
    xy = position[:, 1:3]
    speed = _speed_cm_s(times, xy)
    in_run = _times_in_intervals(times, session.run_times)
    movement = in_run & (speed >= config.min_speed_cm_s)

    x_edges = _fixed_edges(xy[:, 0], config.bin_size_cm, config.n_bins_x)
    y_edges = _fixed_edges(xy[:, 1], config.bin_size_cm, config.n_bins_y)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_x, mesh_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    centers = np.column_stack([mesh_x.reshape(-1), mesh_y.reshape(-1)])
    grid_shape = (config.n_bins_x, config.n_bins_y)
    flat_bins = _positions_to_flat_bins(xy, x_edges, y_edges)
    dt = _frame_durations(times)

    occupancy = np.zeros(config.n_bins_x * config.n_bins_y, dtype=float)
    valid_frames = movement & (flat_bins >= 0)
    np.add.at(occupancy, flat_bins[valid_frames], dt[valid_frames])

    spikes = session.excitatory_spikes() if config.use_excitatory else session.spikes
    cell_ids = (
        np.asarray(session.excitatory_neurons, dtype=int)
        if config.use_excitatory and session.excitatory_neurons.size
        else np.unique(spikes[:, 1].astype(int))
    )
    cell_ids = np.asarray(sorted(np.unique(cell_ids)), dtype=int)
    counts = np.zeros((cell_ids.shape[0], occupancy.shape[0]), dtype=float)
    if spikes.size and cell_ids.size:
        spike_times = spikes[:, 0]
        spike_cell_ids = spikes[:, 1].astype(int)
        spike_xy = _interp_positions(times, xy, spike_times)
        spike_speed = np.interp(spike_times, times, speed)
        spike_in_run = _times_in_intervals(spike_times, session.run_times)
        spike_bins = _positions_to_flat_bins(spike_xy, x_edges, y_edges)
        keep_spikes = spike_in_run & (spike_speed >= config.min_speed_cm_s) & (spike_bins >= 0)
        kept_cell_ids = spike_cell_ids[keep_spikes]
        kept_bins = spike_bins[keep_spikes].astype(int)
        rows = np.searchsorted(cell_ids, kept_cell_ids)
        valid_rows = (rows >= 0) & (rows < cell_ids.shape[0])
        valid_rows[valid_rows] &= cell_ids[rows[valid_rows]] == kept_cell_ids[valid_rows]
        np.add.at(counts, (rows[valid_rows], kept_bins[valid_rows]), 1.0)

    sigma_bins = config.smoothing_sigma_cm / config.bin_size_cm
    occupancy_grid = occupancy.reshape(grid_shape)
    if sigma_bins > 0.0:
        smooth_occupancy = gaussian_filter(occupancy_grid, sigma=sigma_bins, mode="constant").reshape(-1)
        smooth_counts = _smooth_count_rows(counts, grid_shape, sigma_bins)
    else:
        smooth_occupancy = occupancy
        smooth_counts = counts

    denominator = np.maximum(smooth_occupancy, config.min_occupancy_s)
    rates = np.maximum(smooth_counts / denominator[None, :], config.rate_floor_hz)
    if rates.shape[0]:
        keep_cells = np.max(rates, axis=1) >= config.min_peak_rate_hz
        rates = rates[keep_cells]
        cell_ids = cell_ids[keep_cells]
    return EncodingModel(
        x_edges=x_edges,
        y_edges=y_edges,
        bin_centers=centers,
        rates_hz=rates,
        occupancy_s=occupancy,
        cell_ids=cell_ids,
        config=config,  # type: ignore[arg-type]
    )


def build_kd_emissions(
    session: ReplaySession,
    encoding: EncodingModel,
    ripple: RippleEvent | int,
    time_bin_s: float,
    spike_rate_scale: float = 1.0,
) -> LogEmissionTensor:
    ripple_event = _coerce_ripple_event(session, ripple)
    edges = _ripple_time_edges(ripple_event.start, ripple_event.end, time_bin_s)
    bin_durations = np.diff(edges)
    times = edges[:-1] + 0.5 * bin_durations
    dt = float(np.median(bin_durations))
    counts = np.zeros((times.shape[0], encoding.n_cells), dtype=int)
    if session.spikes.size and encoding.n_cells:
        keep = (
            (session.spikes[:, 0] >= edges[0])
            & (session.spikes[:, 0] < edges[-1])
            & np.isin(session.spikes[:, 1].astype(int), encoding.cell_ids)
        )
        spike_times = session.spikes[keep, 0]
        spike_cell_ids = session.spikes[keep, 1].astype(int)
        time_bins = np.searchsorted(edges, spike_times, side="right") - 1
        rows = np.searchsorted(encoding.cell_ids, spike_cell_ids)
        valid = (time_bins >= 0) & (time_bins < counts.shape[0])
        valid &= (rows >= 0) & (rows < encoding.cell_ids.shape[0])
        valid[valid] &= encoding.cell_ids[rows[valid]] == spike_cell_ids[valid]
        np.add.at(counts, (time_bins[valid].astype(int), rows[valid]), 1)
    log_likelihood = poisson_log_emissions(
        counts,
        encoding.rates_hz,
        bin_durations,
        spike_rate_scale=spike_rate_scale,
    )
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=counts,
        times=times,
        dt=dt,
        cell_ids=encoding.cell_ids,
        n_spikes=int(counts.sum()),
        bin_durations=bin_durations,
        transition_durations=np.diff(times) if times.shape[0] > 1 else np.empty(0, dtype=float),
    )


def _ripple_time_edges(start: float, end: float, time_bin_s: float) -> np.ndarray:
    start = float(start)
    end = float(end)
    time_bin_s = float(time_bin_s)
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("ripple start and end must be finite")
    if not np.isfinite(time_bin_s) or time_bin_s <= 0.0:
        raise ValueError("time_bin_s must be finite and positive")
    if end <= start:
        raise ValueError("ripple end must be greater than ripple start")

    duration = end - start
    n_full_bins = int(np.floor(duration / time_bin_s))
    edges = start + np.arange(n_full_bins + 1, dtype=float) * time_bin_s
    tolerance = max(
        np.finfo(float).eps * max(abs(start), abs(end), 1.0) * 16.0,
        time_bin_s * 1e-9,
    )
    if edges[-1] > end and not np.isclose(edges[-1], end, rtol=0.0, atol=tolerance):
        edges = edges[edges < end]
    if not np.isclose(edges[-1], end, rtol=0.0, atol=tolerance):
        edges = np.append(edges, float(end))
    else:
        edges[-1] = float(end)
    return edges


def poisson_log_emissions(
    spike_counts: np.ndarray,
    rates_hz: np.ndarray,
    dt: float | np.ndarray,
    *,
    spike_rate_scale: float = 1.0,
) -> np.ndarray:
    spike_counts, rates_hz = _validate_poisson_inputs(spike_counts, rates_hz)
    dt_array = np.asarray(dt, dtype=float)
    if not np.isfinite(spike_rate_scale) or spike_rate_scale <= 0.0:
        raise ValueError("spike_rate_scale must be finite and positive")
    if dt_array.ndim == 0:
        if not np.isfinite(float(dt_array)) or float(dt_array) <= 0.0:
            raise ValueError("dt must be finite and positive")
        expected = np.maximum(rates_hz * float(dt_array) * spike_rate_scale, np.finfo(float).tiny)
        return spike_counts @ np.log(expected) - expected.sum(axis=0)[None, :] - gammaln(spike_counts + 1).sum(axis=1)[:, None]

    if dt_array.ndim != 1 or dt_array.shape[0] != spike_counts.shape[0]:
        raise ValueError("dt must be a scalar or one duration per time bin")
    if not np.all(np.isfinite(dt_array)) or np.any(dt_array <= 0.0):
        raise ValueError("all bin durations must be finite and positive")

    expected = np.maximum(
        dt_array[:, None, None] * rates_hz[None, :, :] * spike_rate_scale,
        np.finfo(float).tiny,
    )
    return (
        np.einsum("tc,tcb->tb", spike_counts, np.log(expected), optimize=True)
        - expected.sum(axis=1)
        - gammaln(spike_counts + 1).sum(axis=1)[:, None]
    )


def kd_random_log_evidence(log_emissions: np.ndarray) -> float:
    return float(np.sum(logsumexp(log_emissions, axis=1) - np.log(log_emissions.shape[1])))


def kd_stationary_log_evidence(log_emissions: np.ndarray) -> float:
    return float(logsumexp(np.sum(log_emissions, axis=0) - np.log(log_emissions.shape[1])))


def kd_stationary_gaussian_log_evidence(log_emissions: np.ndarray, n_bins_x: int, n_bins_y: int, sd_meters: float, bin_size_cm: float) -> float:
    transition_x = stationary_gaussian_transition_1d(n_bins_x, sd_meters, bin_size_cm)
    transition_y = stationary_gaussian_transition_1d(n_bins_y, sd_meters, bin_size_cm)
    return kd_stationary_gaussian_log_evidence_from_transitions(log_emissions, n_bins_x, n_bins_y, transition_x, transition_y)


def kd_stationary_gaussian_log_evidence_from_transitions(
    log_emissions: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    transition_x: np.ndarray,
    transition_y: np.ndarray | None = None,
) -> float:
    transition_y = transition_x if transition_y is None else transition_y
    log_terms = np.zeros((n_bins_x, n_bins_y), dtype=float)
    for time_index in range(log_emissions.shape[0]):
        emission, offset = _scaled_emission(log_emissions, time_index)
        weighted = transition_x.T @ emission.reshape(n_bins_x, n_bins_y) @ transition_y
        if np.any(weighted <= 0.0):
            weighted = np.maximum(weighted, np.finfo(float).tiny)
        log_terms += np.log(weighted) + offset
    return float(logsumexp(log_terms.reshape(-1) - np.log(log_emissions.shape[1])))


def kd_stationary_gaussian_log_evidence_from_latent(log_emissions: np.ndarray, latent: np.ndarray) -> float:
    log_terms = [logsumexp(log_emissions[t, :, None] + latent, axis=0) for t in range(log_emissions.shape[0])]
    return float(logsumexp(np.sum(log_terms, axis=0) - np.log(log_emissions.shape[1])))


def kd_diffusion_log_evidence(log_emissions: np.ndarray, n_bins_x: int, n_bins_y: int, sd_meters: float, bin_size_cm: float, dt: float) -> float:
    transition = diffusion_transition_1d(n_bins_x, sd_meters, bin_size_cm, dt)
    return kd_diffusion_log_evidence_from_transition(log_emissions, n_bins_x, n_bins_y, transition)


def kd_diffusion_log_evidence_from_transition(log_emissions: np.ndarray, n_bins_x: int, n_bins_y: int, transition: np.ndarray) -> float:
    return _first_order_separable_log_evidence(log_emissions, n_bins_x, n_bins_y, transition)


def kd_momentum_log_evidence(
    log_emissions: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    sd_meters: float,
    decay: float,
    initial_sd_m_per_s: float,
    bin_size_cm: float,
    dt: float,
) -> float:
    if n_bins_x != n_bins_y:
        raise ValueError("KD momentum scorer currently requires a square grid")
    if log_emissions.shape[0] == 1:
        return kd_random_log_evidence(log_emissions)
    if decay > 1.0:
        decay, sd_meters = adjusted_momentum_parameters(decay, sd_meters, dt)
    initial_sd_meters = initial_sd_m_per_s * dt
    initial = diffusion_transition_1d(n_bins_x, initial_sd_meters, bin_size_cm, dt=1.0)
    transition = momentum_transition_1d(n_bins_x, sd_meters, decay, bin_size_cm, dt)
    return kd_momentum_log_evidence_from_transitions(log_emissions, n_bins_x, initial, transition)


def kd_momentum_log_evidence_from_transitions(log_emissions: np.ndarray, n_bins: int, initial: np.ndarray, transition: np.ndarray) -> float:
    if log_emissions.shape[0] == 1:
        return kd_random_log_evidence(log_emissions)
    return _second_order_separable_log_evidence(log_emissions, n_bins, initial, transition)


def stationary_gaussian_log_latent(n_bins_x: int, n_bins_y: int, sd_meters: float, bin_size_cm: float) -> np.ndarray:
    sd_bins = meters_to_bins(sd_meters, bin_size_cm)
    x = np.arange(n_bins_x)
    y = np.arange(n_bins_y)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    states = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    latent = np.empty((n_bins_x * n_bins_y, n_bins_x * n_bins_y), dtype=float)
    for src, center in enumerate(states):
        dist2 = np.sum((states - center[None, :]) ** 2, axis=1)
        log_weights = -0.5 * dist2 / (sd_bins * sd_bins)
        latent[:, src] = log_weights - logsumexp(log_weights)
    return latent


def stationary_gaussian_transition_1d(n_bins: int, sd_meters: float, bin_size_cm: float) -> np.ndarray:
    sd_bins = meters_to_bins(sd_meters, bin_size_cm)
    variance = max(sd_bins * sd_bins, np.finfo(float).tiny)
    return _gaussian_transition_1d(n_bins, variance)


def diffusion_transition_1d(n_bins: int, sd_meters: float, bin_size_cm: float, dt: float) -> np.ndarray:
    sd_bins = meters_to_bins(sd_meters, bin_size_cm)
    variance = max(sd_bins * sd_bins * dt, np.finfo(float).tiny)
    return _gaussian_transition_1d(n_bins, variance)


def _gaussian_transition_1d(n_bins: int, variance: float) -> np.ndarray:
    current = np.arange(n_bins)
    transition = np.empty((n_bins, n_bins), dtype=float)
    for prev in range(n_bins):
        weights = np.exp(-((current - prev) ** 2) / (2.0 * variance))
        total = weights.sum()
        if total == 0.0:
            weights[np.argmin(np.abs(current - prev))] = 1.0
            total = 1.0
        transition[:, prev] = weights / total
    return transition


def momentum_transition_1d(n_bins: int, sd_meters: float, decay: float, bin_size_cm: float, dt: float) -> np.ndarray:
    sd_bins = meters_to_bins(sd_meters, bin_size_cm)
    if decay <= 0.0:
        raise ValueError("momentum decay must be positive")
    var_scaled = (sd_bins * sd_bins * dt * dt) / (2.0 * decay) * (1.0 - np.exp(-2.0 * decay * dt))
    var_scaled = max(float(var_scaled), np.finfo(float).tiny)
    current = np.arange(n_bins)
    transition = np.empty((n_bins, n_bins, n_bins), dtype=float)
    for prev_prev in range(n_bins):
        for prev in range(n_bins):
            mean = (1.0 + np.exp(-dt * decay)) * prev - np.exp(-dt * decay) * prev_prev
            weights = np.exp(-((current - mean) ** 2) / (2.0 * var_scaled))
            total = weights.sum()
            if total == 0.0:
                weights[0 if mean < 0 else n_bins - 1] = 1.0
                total = 1.0
            transition[:, prev, prev_prev] = weights / total
    return transition


def marginalize_grid_log_evidence(grid: np.ndarray, prior: np.ndarray) -> np.ndarray:
    if grid.shape[1:] != prior.shape:
        raise ValueError(f"grid shape {grid.shape[1:]} does not match prior shape {prior.shape}")
    return logsumexp(grid + np.log(np.maximum(prior, np.finfo(float).tiny)), axis=tuple(range(1, grid.ndim)))


def empirical_grid_prior(grid_params: dict[str, np.ndarray], grid: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if grid.ndim == 2:
        best = grid_params["sd_meters"][np.nanargmax(grid, axis=1)]
        prior = _fit_invgamma_prior(best, grid_params["sd_meters"])
        return prior, {"sd_prior_mass": float(prior.sum())}
    if grid.ndim == 3:
        flat = np.nanargmax(grid.reshape(grid.shape[0], -1), axis=1)
        sd_idx, decay_idx = np.unravel_index(flat, grid.shape[1:])
        prior = _fit_lognormal2d_prior(
            grid_params["sd_meters"][sd_idx],
            grid_params["decay"][decay_idx],
            grid_params["sd_meters"],
            grid_params["decay"],
        )
        return prior, {"joint_prior_mass": float(prior.sum())}
    raise ValueError("grid must have 2 or 3 dimensions")


def best_grid_params(model: str, event_indices: Iterable[int], grid_params: dict[str, np.ndarray], grid: np.ndarray) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for row_index, event_index in enumerate(event_indices):
        if grid.ndim == 2:
            best = int(np.nanargmax(grid[row_index]))
            rows.append(
                {
                    "event_index": int(event_index),
                    "model": model,
                    "best_sd_meters": float(grid_params["sd_meters"][best]),
                    "best_log_evidence": float(grid[row_index, best]),
                }
            )
        else:
            best = int(np.nanargmax(grid[row_index].reshape(-1)))
            sd_idx, decay_idx = np.unravel_index(best, grid.shape[1:])
            rows.append(
                {
                    "event_index": int(event_index),
                    "model": model,
                    "best_sd_meters": float(grid_params["sd_meters"][sd_idx]),
                    "best_decay": float(grid_params["decay"][decay_idx]),
                    "best_log_evidence": float(grid[row_index, sd_idx, decay_idx]),
                }
            )
    return rows


def random_effects_model_probabilities(log_evidence: np.ndarray, models: list[str], prior: float = 10.0, n_iterations: int = 500, burnin: int = 50) -> list[dict[str, float | str]]:
    finite = log_evidence[np.all(np.isfinite(log_evidence), axis=1)]
    if finite.size == 0:
        return [{"model": model, "p_model": np.nan, "p_exceedance": np.nan} for model in models]
    rng = np.random.default_rng(0)
    centered = finite - finite.max(axis=1, keepdims=True)
    n_events, n_models = centered.shape
    gibbs = np.zeros((n_iterations, n_models), dtype=float)
    alpha = np.ones(n_models, dtype=float) * prior
    for iteration in range(n_iterations):
        r_m = rng.dirichlet(alpha)
        gibbs[iteration] = r_m
        log_assignment = centered + np.log(r_m)[None, :]
        assignment_probs = np.exp(log_assignment - logsumexp(log_assignment, axis=1, keepdims=True))
        counts = np.zeros(n_models, dtype=int)
        for event_index in range(n_events):
            counts += rng.multinomial(1, assignment_probs[event_index])
        alpha = prior + counts
    posterior = gibbs[burnin:]
    p_models = posterior.mean(axis=0)
    p_exceedance = (posterior == posterior.max(axis=1, keepdims=True)).mean(axis=0)
    return [
        {
            "model": model,
            "p_model": float(p_models[index]),
            "p_exceedance": float(p_exceedance[index]),
        }
        for index, model in enumerate(models)
    ]


def meters_to_bins(value_meters: float | np.ndarray, bin_size_cm: float) -> float | np.ndarray:
    converted = np.asarray(value_meters) * 100.0 / bin_size_cm
    return float(converted) if converted.shape == () else converted


def adjusted_momentum_parameters(theta: float, sigma: float, delta_t: float) -> tuple[float, float]:
    n = np.power(10, 10)
    theta_adjusted = np.log(delta_t * theta + 1.0) / delta_t
    delta = n * delta_t
    continuous_function = (
        sigma**2
        / theta
        * ((2.0 * theta * delta) - np.exp(-2.0 * theta * delta) + 4.0 * np.exp(-theta * delta) - 3.0)
        / (2.0 * theta**2)
    )
    prefactor = delta_t**2 / (2.0 * theta_adjusted)
    numerator = (
        (delta / delta_t) * (-np.exp(2.0 * theta_adjusted * delta_t))
        - 2.0 * np.exp(-theta_adjusted * (delta - delta_t))
        - 2.0 * np.exp(-theta_adjusted * delta)
        + np.exp(-2.0 * theta_adjusted * delta)
        + 2.0 * np.exp(theta_adjusted * delta_t)
        + (delta / delta_t)
        + 1.0
    )
    denominator = (np.exp(theta_adjusted * delta_t) - 1.0) ** 2
    discrete_function = prefactor * -(numerator / denominator)
    sigma_adjusted = np.sqrt(continuous_function / discrete_function)
    return float(theta_adjusted), float(sigma_adjusted)


def _fixed_edges(values: np.ndarray, bin_size_cm: float, n_bins: int) -> np.ndarray:
    low = np.floor(float(np.nanmin(values)) / bin_size_cm) * bin_size_cm
    high = low + n_bins * bin_size_cm
    max_value = float(np.nanmax(values))
    if max_value > high:
        high = np.ceil(max_value / bin_size_cm) * bin_size_cm
        low = high - n_bins * bin_size_cm
    return low + np.arange(n_bins + 1, dtype=float) * bin_size_cm


def _scaled_emission(log_emissions: np.ndarray, time_index: int) -> tuple[np.ndarray, float]:
    row = log_emissions[time_index]
    offset = float(np.max(row))
    return np.exp(row - offset), offset


def _first_order_separable_log_evidence(log_emissions: np.ndarray, n_bins_x: int, n_bins_y: int, transition: np.ndarray) -> float:
    emission, offset = _scaled_emission(log_emissions, 0)
    alpha = emission.reshape(n_bins_x, n_bins_y) / log_emissions.shape[1]
    conditional = float(alpha.sum())
    logp = np.log(conditional) + offset
    alpha /= conditional
    for time_index in range(1, log_emissions.shape[0]):
        emission, offset = _scaled_emission(log_emissions, time_index)
        predicted = transition @ alpha @ transition.T
        alpha = predicted * emission.reshape(n_bins_x, n_bins_y)
        conditional = float(alpha.sum())
        if conditional <= 0.0:
            return float("-inf")
        logp += np.log(conditional) + offset
        alpha /= conditional
    return float(logp)


def _second_order_separable_log_evidence(log_emissions: np.ndarray, n_bins: int, initial: np.ndarray, transition: np.ndarray) -> float:
    emission0, offset0 = _scaled_emission(log_emissions, 0)
    alpha0 = emission0.reshape(n_bins, n_bins) / log_emissions.shape[1]
    conditional0 = float(alpha0.sum())
    logp = np.log(conditional0) + offset0
    alpha0 /= conditional0

    emission1, offset1 = _scaled_emission(log_emissions, 1)
    emission1_grid = emission1.reshape(n_bins, n_bins)
    alpha_t = np.einsum("ip,jq,pq,ij->ijpq", initial, initial, alpha0, emission1_grid, optimize=True)
    conditional1 = float(alpha_t.sum())
    if conditional1 <= 0.0:
        return float("-inf")
    logp += np.log(conditional1) + offset1
    alpha_t /= conditional1

    for time_index in range(2, log_emissions.shape[0]):
        emission, offset = _scaled_emission(log_emissions, time_index)
        emission_grid = emission.reshape(n_bins, n_bins)
        y_sum = np.einsum("jbq,abpq->abpj", transition, alpha_t, optimize=True)
        predicted = np.einsum("iap,abpj->ijab", transition, y_sum, optimize=True)
        alpha_t = predicted * emission_grid[:, :, None, None]
        conditional = float(alpha_t.sum())
        if conditional <= 0.0:
            return float("-inf")
        logp += np.log(conditional) + offset
        alpha_t /= conditional
    return float(logp)


def _fit_invgamma_prior(best_values: np.ndarray, grid_values: np.ndarray) -> np.ndarray:
    interior = best_values[(best_values != grid_values[0]) & (best_values != grid_values[-1])]
    if interior.shape[0] < 3 or np.allclose(interior, interior[0]):
        return np.ones(grid_values.shape, dtype=float) / grid_values.shape[0]
    try:
        a, loc, scale = invgamma.fit(interior)
        prior = invgamma.pdf(grid_values, a=a, loc=loc, scale=scale)
    except Exception:
        prior = np.ones(grid_values.shape, dtype=float)
    if not np.all(np.isfinite(prior)) or prior.sum() <= 0.0:
        prior = np.ones(grid_values.shape, dtype=float)
    return prior / prior.sum()


def _fit_lognormal2d_prior(best_sd: np.ndarray, best_decay: np.ndarray, sd_grid: np.ndarray, decay_grid: np.ndarray) -> np.ndarray:
    interior = (best_sd != sd_grid[0]) & (best_sd != sd_grid[-1]) & (best_decay != decay_grid[0]) & (best_decay != decay_grid[-1])
    if interior.sum() < 3:
        return np.ones((sd_grid.shape[0], decay_grid.shape[0]), dtype=float) / (sd_grid.shape[0] * decay_grid.shape[0])
    data = np.vstack((np.log(best_sd[interior]), np.log(best_decay[interior])))
    try:
        mean = np.mean(data, axis=1)
        cov = np.cov(data) + np.eye(2) * 1e-9
        decay_mesh, sd_mesh = np.meshgrid(decay_grid, sd_grid)
        prior = multivariate_normal.pdf(np.dstack((np.log(sd_mesh), np.log(decay_mesh))), mean=mean, cov=cov)
    except Exception:
        prior = np.ones((sd_grid.shape[0], decay_grid.shape[0]), dtype=float)
    if not np.all(np.isfinite(prior)) or prior.sum() <= 0.0:
        prior = np.ones((sd_grid.shape[0], decay_grid.shape[0]), dtype=float)
    return prior / prior.sum()
