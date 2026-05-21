"""Poisson place-field encoding and spike-emission construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln

from .data import ReplaySession, RippleEvent


@dataclass(frozen=True)
class EncodingConfig:
    bin_size_cm: float = 4.0
    smoothing_sigma_bins: float = 1.5
    min_speed_cm_s: float = 5.0
    min_occupancy_s: float = 0.02
    rate_floor_hz: float = 1e-4
    arena_padding_cm: float = 2.0
    use_excitatory: bool = True
    exclude_ripple_intervals: bool = True


@dataclass(frozen=True)
class EmissionConfig:
    time_bin_s: float = 0.02
    spike_rate_scale: float = 1.0
    likelihood_temperature: float = 1.0
    cell_weights: Iterable[float] | np.ndarray | None = None
    negative_binomial_overdispersion: float = 0.0


@dataclass
class EncodingModel:
    """Occupancy-normalized place-field encoding model."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    bin_centers: np.ndarray
    rates_hz: np.ndarray
    occupancy_s: np.ndarray
    cell_ids: np.ndarray
    config: EncodingConfig

    @property
    def n_bins(self) -> int:
        return int(self.bin_centers.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cell_ids.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (len(self.x_edges) - 1, len(self.y_edges) - 1)

    def select_cells(self, cell_ids: Iterable[int]) -> "EncodingModel":
        requested = np.asarray(sorted(set(cell_ids)), dtype=int)
        indices: list[int] = []
        missing: list[int] = []
        for cell_id in requested:
            matches = np.flatnonzero(self.cell_ids == cell_id)
            if matches.size:
                indices.append(int(matches[0]))
            else:
                missing.append(int(cell_id))

        if missing:
            raise ValueError(
                "requested cell IDs are not present in encoding model: "
                f"{missing}; available cell IDs: {self.cell_ids.astype(int).tolist()}"
            )

        return EncodingModel(
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            bin_centers=self.bin_centers,
            rates_hz=self.rates_hz[np.asarray(indices, dtype=int)],
            occupancy_s=self.occupancy_s,
            cell_ids=requested,
            config=self.config,
        )

    def positions_to_flat_bins(self, xy: np.ndarray) -> np.ndarray:
        x_idx = np.searchsorted(self.x_edges, xy[:, 0], side="right") - 1
        y_idx = np.searchsorted(self.y_edges, xy[:, 1], side="right") - 1
        valid = (
            (x_idx >= 0)
            & (x_idx < self.grid_shape[0])
            & (y_idx >= 0)
            & (y_idx < self.grid_shape[1])
        )
        flat = np.full(x_idx.shape, -1, dtype=int)
        flat[valid] = x_idx[valid] * self.grid_shape[1] + y_idx[valid]
        return flat


@dataclass
class LogEmissionTensor:
    """Per-time-bin log likelihood of observed spikes for every spatial bin."""

    log_likelihood: np.ndarray
    spike_counts: np.ndarray
    times: np.ndarray
    dt: float
    cell_ids: np.ndarray
    n_spikes: int
    bin_durations: np.ndarray | None = None
    transition_durations: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        return int(self.log_likelihood.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.log_likelihood.shape[1])

    def __post_init__(self) -> None:
        """Attach explicit bin/transition durations for duration-aware dynamics."""

        if self.bin_durations is None:
            self.bin_durations = np.full(self.n_time, float(self.dt), dtype=float)
        else:
            self.bin_durations = np.asarray(self.bin_durations, dtype=float)
            if self.bin_durations.shape != (self.n_time,):
                raise ValueError("bin_durations must contain one duration per emission row")
        if self.transition_durations is None:
            if self.n_time <= 1:
                self.transition_durations = np.empty(0, dtype=float)
            elif self.times.shape == (self.n_time,):
                values = np.diff(np.asarray(self.times, dtype=float))
                self.transition_durations = values if np.all(values > 0.0) else np.full(self.n_time - 1, float(self.dt), dtype=float)
            elif self.bin_durations is not None:
                self.transition_durations = 0.5 * (self.bin_durations[:-1] + self.bin_durations[1:])
            else:
                self.transition_durations = np.full(self.n_time - 1, float(self.dt), dtype=float)
        else:
            self.transition_durations = np.asarray(self.transition_durations, dtype=float)
            if self.transition_durations.shape != (max(self.n_time - 1, 0),):
                raise ValueError("transition_durations must contain one duration per adjacent time-bin pair")


def fit_place_field_encoding(session: ReplaySession, config: EncodingConfig | None = None) -> EncodingModel:
    """Fit occupancy-normalized Poisson rates from non-replay movement periods."""

    config = EncodingConfig() if config is None else config
    position = _clean_position(session.position)
    selected_spikes = session.excitatory_spikes() if config.use_excitatory else session.spikes
    if not (position.shape[0] == 1 and np.asarray(selected_spikes).size == 0):
        _validate_position_samples(position)
    times = position[:, 0]
    xy = position[:, 1:3]
    speed = _speed_cm_s(times, xy)
    in_run = _times_in_intervals(times, session.run_times)
    excluded_intervals = _encoding_exclusion_intervals(session, config)
    in_excluded_interval = _times_in_intervals(times, excluded_intervals)
    movement = in_run & ~in_excluded_interval & (speed >= config.min_speed_cm_s)

    x_edges, y_edges, centers = _make_grid(xy, config)
    grid_shape = (len(x_edges) - 1, len(y_edges) - 1)
    flat_bins = _positions_to_flat_bins(xy, x_edges, y_edges)
    dt = _frame_durations(times)

    occupancy = np.zeros(grid_shape[0] * grid_shape[1], dtype=float)
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
        spike_in_excluded_interval = _times_in_intervals(spike_times, excluded_intervals)
        spike_bins = _positions_to_flat_bins(spike_xy, x_edges, y_edges)
        keep_spikes = (
            spike_in_run
            & ~spike_in_excluded_interval
            & (spike_speed >= config.min_speed_cm_s)
            & (spike_bins >= 0)
        )
        kept_cell_ids = spike_cell_ids[keep_spikes]
        kept_bins = spike_bins[keep_spikes].astype(int)
        rows = np.searchsorted(cell_ids, kept_cell_ids)
        valid_rows = (rows >= 0) & (rows < cell_ids.shape[0])
        valid_rows[valid_rows] &= cell_ids[rows[valid_rows]] == kept_cell_ids[valid_rows]
        np.add.at(counts, (rows[valid_rows], kept_bins[valid_rows]), 1.0)

    occupancy_grid = occupancy.reshape(grid_shape)
    if config.smoothing_sigma_bins > 0.0:
        smooth_occupancy = gaussian_filter(
            occupancy_grid,
            sigma=config.smoothing_sigma_bins,
            mode="constant",
        ).reshape(-1)
        smooth_counts = _smooth_count_rows(
            counts,
            grid_shape,
            config.smoothing_sigma_bins,
        )
    else:
        smooth_occupancy = occupancy
        smooth_counts = counts

    denominator = np.maximum(smooth_occupancy, config.min_occupancy_s)
    rates = smooth_counts / denominator[None, :]
    rates = np.maximum(rates, config.rate_floor_hz)
    return EncodingModel(
        x_edges=x_edges,
        y_edges=y_edges,
        bin_centers=centers,
        rates_hz=rates,
        occupancy_s=occupancy,
        cell_ids=cell_ids,
        config=config,
    )


def build_emissions(
    session: ReplaySession,
    encoding: EncodingModel,
    ripple: RippleEvent | int,
    config: EmissionConfig | None = None,
) -> LogEmissionTensor:
    """Build Poisson log-emission likelihoods for one ripple."""

    config = EmissionConfig() if config is None else config
    ripple_event = session.ripple(ripple) if isinstance(ripple, int) else ripple
    edges = _time_bin_edges(ripple_event.start, ripple_event.end, config.time_bin_s)
    bin_durations = np.diff(edges)
    times = edges[:-1] + 0.5 * bin_durations
    dt = float(np.median(bin_durations))
    spikes = session.spikes
    counts = np.zeros((times.shape[0], encoding.n_cells), dtype=int)
    if spikes.size and encoding.n_cells:
        keep = (
            (spikes[:, 0] >= ripple_event.start)
            & (spikes[:, 0] < ripple_event.end)
            & np.isin(spikes[:, 1].astype(int), encoding.cell_ids)
        )
        spike_times = spikes[keep, 0]
        spike_cell_ids = spikes[keep, 1].astype(int)
        time_bins = np.searchsorted(edges, spike_times, side="right") - 1
        rows = np.searchsorted(encoding.cell_ids, spike_cell_ids)
        valid = (time_bins >= 0) & (time_bins < counts.shape[0])
        valid &= (rows >= 0) & (rows < encoding.cell_ids.shape[0])
        valid[valid] &= encoding.cell_ids[rows[valid]] == spike_cell_ids[valid]
        np.add.at(counts, (time_bins[valid].astype(int), rows[valid]), 1)

    log_likelihood = _poisson_log_emissions(
        counts,
        encoding.rates_hz,
        bin_durations,
        spike_rate_scale=config.spike_rate_scale,
        likelihood_temperature=config.likelihood_temperature,
        cell_weights=config.cell_weights,
        negative_binomial_overdispersion=config.negative_binomial_overdispersion,
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


def _poisson_log_emissions(
    spike_counts: np.ndarray,
    rates_hz: np.ndarray,
    dt: float | np.ndarray,
    *,
    spike_rate_scale: float = 1.0,
    likelihood_temperature: float = 1.0,
    cell_weights: Iterable[float] | np.ndarray | None = None,
    negative_binomial_overdispersion: float = 0.0,
) -> np.ndarray:
    if not np.isfinite(spike_rate_scale) or spike_rate_scale <= 0.0:
        raise ValueError("spike_rate_scale must be finite and positive")
    _validate_emission_calibration(
        likelihood_temperature=likelihood_temperature,
        negative_binomial_overdispersion=negative_binomial_overdispersion,
    )
    weights = _emission_cell_weights(cell_weights, spike_counts.shape[1])
    dt_array = np.asarray(dt, dtype=float)
    if dt_array.ndim == 0:
        if float(dt_array) <= 0.0:
            raise ValueError("dt must be positive")
        expected = np.maximum(rates_hz * float(dt_array) * spike_rate_scale, np.finfo(float).tiny)
        log_likelihood = _count_log_emissions(
            spike_counts,
            expected,
            cell_weights=weights,
            negative_binomial_overdispersion=negative_binomial_overdispersion,
        )
        return _apply_likelihood_temperature(log_likelihood, likelihood_temperature)

    if dt_array.ndim != 1 or dt_array.shape[0] != spike_counts.shape[0]:
        raise ValueError("dt must be a scalar or one duration per time bin")
    if np.any(dt_array <= 0.0):
        raise ValueError("all bin durations must be positive")

    expected = np.maximum(
        dt_array[:, None, None] * rates_hz[None, :, :] * spike_rate_scale,
        np.finfo(float).tiny,
    )
    log_likelihood = _count_log_emissions(
        spike_counts,
        expected,
        cell_weights=weights,
        negative_binomial_overdispersion=negative_binomial_overdispersion,
    )
    return _apply_likelihood_temperature(log_likelihood, likelihood_temperature)


def _count_log_emissions(
    spike_counts: np.ndarray,
    expected: np.ndarray,
    *,
    cell_weights: np.ndarray,
    negative_binomial_overdispersion: float,
) -> np.ndarray:
    """Return per-time/bin count log likelihoods for Poisson or NB emissions."""

    if negative_binomial_overdispersion == 0.0:
        if expected.ndim == 2:
            return (
                (spike_counts * cell_weights[None, :]) @ np.log(expected)
                - (cell_weights @ expected)[None, :]
                - (gammaln(spike_counts + 1) * cell_weights[None, :]).sum(axis=1)[:, None]
            )
        if expected.ndim == 3:
            return (
                np.einsum(
                    "tc,tcb,c->tb",
                    spike_counts,
                    np.log(expected),
                    cell_weights,
                    optimize=True,
                )
                - np.einsum("tcb,c->tb", expected, cell_weights, optimize=True)
                - (gammaln(spike_counts + 1) * cell_weights[None, :]).sum(axis=1)[:, None]
            )
        raise ValueError("expected must be two- or three-dimensional")

    counts = np.asarray(spike_counts, dtype=float)[:, :, None]
    expected_3d = expected[None, :, :] if expected.ndim == 2 else expected
    log_terms = _negative_binomial_log_emissions(
        counts,
        expected_3d,
        negative_binomial_overdispersion,
    )
    return (log_terms * cell_weights[None, :, None]).sum(axis=1)


def _emission_cell_weights(
    cell_weights: Iterable[float] | np.ndarray | None,
    n_cells: int,
) -> np.ndarray:
    """Return validated nonnegative per-cell emission weights."""

    if cell_weights is None:
        return np.ones(int(n_cells), dtype=float)
    if np.isscalar(cell_weights):
        values = [cell_weights]
    else:
        values = list(cell_weights)
    weights = np.asarray(values, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != int(n_cells):
        raise ValueError("cell_weights must contain one weight per encoded cell")
    if not np.all(np.isfinite(weights)):
        raise ValueError("cell_weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("cell_weights must be nonnegative")
    if not np.any(weights > 0.0):
        raise ValueError("cell_weights must include at least one positive weight")
    return weights


def _negative_binomial_log_emissions(
    counts: np.ndarray,
    expected: np.ndarray,
    negative_binomial_overdispersion: float,
) -> np.ndarray:
    """Negative-binomial log PMF with variance mean + alpha * mean**2."""

    size = 1.0 / float(negative_binomial_overdispersion)
    mean = np.maximum(np.asarray(expected, dtype=float), np.finfo(float).tiny)
    return (
        gammaln(counts + size)
        - gammaln(size)
        - gammaln(counts + 1.0)
        + size * (np.log(size) - np.log(size + mean))
        + counts * (np.log(mean) - np.log(size + mean))
    )


def _validate_emission_calibration(
    *,
    likelihood_temperature: float,
    negative_binomial_overdispersion: float,
) -> None:
    if not np.isfinite(likelihood_temperature) or likelihood_temperature <= 0.0:
        raise ValueError("likelihood_temperature must be finite and positive")
    if not np.isfinite(negative_binomial_overdispersion) or negative_binomial_overdispersion < 0.0:
        raise ValueError("negative_binomial_overdispersion must be finite and nonnegative")


def _apply_likelihood_temperature(log_likelihood: np.ndarray, likelihood_temperature: float) -> np.ndarray:
    return np.asarray(log_likelihood, dtype=float) / float(likelihood_temperature)


def _time_bin_edges(start: float, end: float, time_bin_s: float) -> np.ndarray:
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
    n_complete = int(np.floor(duration / time_bin_s))
    edges = start + np.arange(n_complete + 1, dtype=float) * time_bin_s
    tolerance = 16.0 * np.finfo(float).eps * max(abs(start), abs(end), abs(duration), 1.0)
    if edges.shape[0] == 1 or edges[-1] < end - tolerance:
        edges = np.append(edges, end)
    else:
        edges[-1] = end
    return edges


def _clean_position(position: np.ndarray) -> np.ndarray:
    arr = np.asarray(position, dtype=float)
    keep = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    return arr[keep]


def _validate_position_samples(position: np.ndarray) -> None:
    if position.shape[0] < 2:
        raise ValueError(
            "at least two finite position samples are required to fit place fields"
        )


def _make_grid(xy: np.ndarray, config: EncodingConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, y_min = np.nanmin(xy, axis=0) - config.arena_padding_cm
    x_max, y_max = np.nanmax(xy, axis=0) + config.arena_padding_cm
    x_edges = np.arange(x_min, x_max + config.bin_size_cm, config.bin_size_cm)
    y_edges = np.arange(y_min, y_max + config.bin_size_cm, config.bin_size_cm)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    mesh_x, mesh_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    centers = np.column_stack([mesh_x.reshape(-1), mesh_y.reshape(-1)])
    return x_edges, y_edges, centers


def _positions_to_flat_bins(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    x_idx = np.searchsorted(x_edges, xy[:, 0], side="right") - 1
    y_idx = np.searchsorted(y_edges, xy[:, 1], side="right") - 1
    valid = (x_idx >= 0) & (x_idx < len(x_edges) - 1) & (y_idx >= 0) & (y_idx < len(y_edges) - 1)
    flat = np.full(x_idx.shape, -1, dtype=int)
    flat[valid] = x_idx[valid] * (len(y_edges) - 1) + y_idx[valid]
    return flat


def _smooth_count_rows(
    counts: np.ndarray,
    grid_shape: tuple[int, int],
    sigma: float,
) -> np.ndarray:
    """Apply spatial Gaussian smoothing to each cell's count map."""

    if counts.shape[0] == 0:
        return counts.copy()
    return np.vstack(
        [
            gaussian_filter(
                row.reshape(grid_shape),
                sigma=sigma,
                mode="constant",
            ).reshape(-1)
            for row in counts
        ]
    )


def _speed_cm_s(times: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if times.shape[0] < 2:
        return np.zeros(times.shape, dtype=float)
    dt = np.gradient(times)
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.sqrt(dx * dx + dy * dy) / dt
    return np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)


def _frame_durations(times: np.ndarray) -> np.ndarray:
    if times.shape[0] == 0:
        return np.empty_like(times, dtype=float)
    durations = np.empty_like(times, dtype=float)
    diffs = np.diff(times)
    median = float(np.median(diffs)) if diffs.size else 1.0 / 30.0
    durations[:-1] = diffs if diffs.size else median
    durations[-1] = median
    return np.clip(durations, 0.0, 1.0)


def _interp_positions(times: np.ndarray, xy: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(query_times, times, xy[:, 0]),
            np.interp(query_times, times, xy[:, 1]),
        ]
    )


def _times_in_intervals(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times <= end)
    return mask


def _encoding_exclusion_intervals(session: ReplaySession, config: EncodingConfig) -> np.ndarray:
    if not config.exclude_ripple_intervals:
        return np.empty((0, 2), dtype=float)
    return _ripple_intervals(session)


def _ripple_intervals(session: ReplaySession) -> np.ndarray:
    """Return finite, positive-duration ripple intervals as start/end pairs."""

    events = np.asarray(session.ripple_events, dtype=float)
    if events.size == 0:
        return np.empty((0, 2), dtype=float)
    events = np.atleast_2d(events)
    intervals = events[:, :2]
    finite = np.isfinite(intervals).all(axis=1)
    positive_duration = intervals[:, 1] > intervals[:, 0]
    return np.asarray(intervals[finite & positive_duration], dtype=float)
