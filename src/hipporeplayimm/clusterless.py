"""Clusterless marked-point-process emissions for replay state-space models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln

from .data import ReplaySession, RippleEvent
from .encoding import (
    EmissionConfig,
    EncodingConfig,
    LogEmissionTensor,
    _clean_position,
    _encoding_exclusion_intervals,
    _frame_durations,
    _interp_positions,
    _make_grid,
    _positions_to_flat_bins,
    _speed_cm_s,
    _time_bin_edges,
    _times_in_intervals,
)
from .models import EventScore
from .state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


@dataclass(frozen=True)
class ClusterlessMarkConfig:
    """Configuration for Gaussian marked-point-process encoding."""

    encoding: EncodingConfig | None = None
    mark_smoothing_sigma_bins: float = 1.0
    mark_prior_count: float = 1.0
    mark_variance_floor: float = 1.0
    rate_floor_hz: float = 1e-4
    use_excitatory: bool = True


@dataclass
class ClusterlessMarkEncoding:
    """Position-dependent spike intensity and mark likelihood parameters."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    bin_centers: np.ndarray
    rate_hz: np.ndarray
    occupancy_s: np.ndarray
    effective_spike_count: np.ndarray
    mark_mean: np.ndarray
    mark_variance: np.ndarray
    mark_feature_names: tuple[str, ...]
    spike_mark_source: str
    config: ClusterlessMarkConfig

    @property
    def n_bins(self) -> int:
        return int(self.bin_centers.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.mark_mean.shape[1])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (len(self.x_edges) - 1, len(self.y_edges) - 1)

    def log_mark_likelihood(self, marks: np.ndarray) -> np.ndarray:
        marks = np.asarray(marks, dtype=float)
        if marks.ndim == 1:
            marks = marks[None, :]
        if marks.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} mark features, got {marks.shape[1]}")
        diff = marks[:, None, :] - self.mark_mean[None, :, :]
        log_norm = np.sum(np.log(2.0 * np.pi * self.mark_variance), axis=1)
        quad = np.sum(diff * diff / self.mark_variance[None, :, :], axis=2)
        return -0.5 * (quad + log_norm[None, :])


@dataclass
class ClusterlessStateSpaceReplayModel(StateSpaceReplayModel):
    """State-space replay model using clusterless marked-point-process emissions."""

    mode: str = "diffusion"
    config: StateSpaceDecoderConfig | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name is None or self.name.startswith("state-space-"):
            self.name = f"clusterless-state-space-{self.mode}"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray] | None = None,
    ) -> EventScore:
        score = super().score(emissions, bin_centers, candidate_indices=candidate_indices)
        score.model_name = str(self.name)
        score.diagnostics["state_space_observation_model"] = "clusterless-marked-point-process"
        score.diagnostics["clusterless_mark_likelihood"] = "diagonal-gaussian"
        return score


def fit_clusterless_mark_encoding(
    session: ReplaySession,
    config: ClusterlessMarkConfig | None = None,
) -> ClusterlessMarkEncoding:
    """Fit clusterless marked-point-process parameters from behavioral run periods."""

    config = ClusterlessMarkConfig() if config is None else config
    encoding_config = EncodingConfig() if config.encoding is None else config.encoding
    marks = session.spike_marks
    if marks is None or marks.n_features == 0:
        raise ValueError("Session does not contain spike marks for clusterless encoding.")

    position = _clean_position(session.position)
    times = position[:, 0]
    xy = position[:, 1:3]
    speed = _speed_cm_s(times, xy)
    in_run = _times_in_intervals(times, session.run_times)
    excluded_intervals = _encoding_exclusion_intervals(session, encoding_config)
    in_excluded_interval = _times_in_intervals(times, excluded_intervals)
    movement = in_run & ~in_excluded_interval & (speed >= encoding_config.min_speed_cm_s)

    x_edges, y_edges, centers = _make_grid(xy, encoding_config)
    grid_shape = (len(x_edges) - 1, len(y_edges) - 1)
    flat_bins = _positions_to_flat_bins(xy, x_edges, y_edges)
    dt = _frame_durations(times)
    occupancy = np.zeros(grid_shape[0] * grid_shape[1], dtype=float)
    valid_frames = movement & (flat_bins >= 0)
    np.add.at(occupancy, flat_bins[valid_frames], dt[valid_frames])

    mark_times, mark_values = _training_marks(session, encoding_config, config)
    mark_xy = _interp_positions(times, xy, mark_times)
    mark_speed = np.interp(mark_times, times, speed)
    mark_bins = _positions_to_flat_bins(mark_xy, x_edges, y_edges)
    mark_in_run = _times_in_intervals(mark_times, session.run_times)
    mark_in_excluded_interval = _times_in_intervals(mark_times, excluded_intervals)
    keep = (
        mark_in_run
        & ~mark_in_excluded_interval
        & (mark_speed >= encoding_config.min_speed_cm_s)
        & (mark_bins >= 0)
    )
    mark_times = mark_times[keep]
    mark_values = mark_values[keep]
    mark_bins = mark_bins[keep].astype(int)
    finite_rows = np.all(np.isfinite(mark_values), axis=1)
    mark_values = mark_values[finite_rows]
    mark_bins = mark_bins[finite_rows]
    if mark_values.shape[0] == 0:
        raise ValueError("No finite run-period spike marks were available for clusterless encoding.")

    raw_count = np.zeros(occupancy.shape[0], dtype=float)
    mark_sum = np.zeros((mark_values.shape[1], occupancy.shape[0]), dtype=float)
    mark_sq_sum = np.zeros_like(mark_sum)
    np.add.at(raw_count, mark_bins, 1.0)
    for feature_index in range(mark_values.shape[1]):
        feature = mark_values[:, feature_index]
        np.add.at(mark_sum[feature_index], mark_bins, feature)
        np.add.at(mark_sq_sum[feature_index], mark_bins, feature * feature)

    sigma = float(config.mark_smoothing_sigma_bins)
    occupancy_grid = occupancy.reshape(grid_shape)
    count_grid = raw_count.reshape(grid_shape)
    if encoding_config.smoothing_sigma_bins > 0.0:
        smooth_occupancy = gaussian_filter(
            occupancy_grid,
            sigma=encoding_config.smoothing_sigma_bins,
            mode="constant",
        ).reshape(-1)
    else:
        smooth_occupancy = occupancy
    if sigma > 0.0:
        smooth_count = gaussian_filter(count_grid, sigma=sigma, mode="constant").reshape(-1)
        smooth_sum = np.vstack(
            [gaussian_filter(row.reshape(grid_shape), sigma=sigma, mode="constant").reshape(-1) for row in mark_sum]
        )
        smooth_sq_sum = np.vstack(
            [gaussian_filter(row.reshape(grid_shape), sigma=sigma, mode="constant").reshape(-1) for row in mark_sq_sum]
        )
    else:
        smooth_count = raw_count
        smooth_sum = mark_sum
        smooth_sq_sum = mark_sq_sum

    global_mean = np.mean(mark_values, axis=0)
    global_variance = np.var(mark_values, axis=0)
    global_variance = np.maximum(global_variance, config.mark_variance_floor)
    prior = max(float(config.mark_prior_count), 0.0)
    denominator = smooth_count + prior
    safe_denominator = np.maximum(denominator, np.finfo(float).tiny)
    prior_second_moment = global_variance + global_mean * global_mean
    mean = (smooth_sum.T + prior * global_mean[None, :]) / safe_denominator[:, None]
    second_moment = (smooth_sq_sum.T + prior * prior_second_moment[None, :]) / safe_denominator[:, None]
    variance = np.maximum(second_moment - mean * mean, config.mark_variance_floor)
    no_support = denominator <= np.finfo(float).tiny
    if np.any(no_support):
        mean[no_support] = global_mean
        variance[no_support] = global_variance

    occupancy_denominator = np.maximum(smooth_occupancy, encoding_config.min_occupancy_s)
    rate_hz = np.maximum(smooth_count / occupancy_denominator, config.rate_floor_hz)
    return ClusterlessMarkEncoding(
        x_edges=x_edges,
        y_edges=y_edges,
        bin_centers=centers,
        rate_hz=rate_hz,
        occupancy_s=occupancy,
        effective_spike_count=smooth_count,
        mark_mean=mean,
        mark_variance=variance,
        mark_feature_names=marks.feature_names,
        spike_mark_source=f"{marks.source_file}:{marks.source_variable}",
        config=config,
    )


def build_clusterless_mark_emissions(
    session: ReplaySession,
    encoding: ClusterlessMarkEncoding,
    ripple: RippleEvent | int,
    config: EmissionConfig | None = None,
) -> LogEmissionTensor:
    """Build marked-point-process log emissions for one ripple."""

    config = EmissionConfig() if config is None else config
    marks = session.spike_marks
    if marks is None or marks.n_features == 0:
        raise ValueError("Session does not contain spike marks for clusterless emission scoring.")
    ripple_event = session.ripple(ripple) if isinstance(ripple, int) else ripple
    edges = _time_bin_edges(ripple_event.start, ripple_event.end, config.time_bin_s)
    bin_durations = np.diff(edges)
    times = edges[:-1] + 0.5 * bin_durations
    dt = float(np.median(bin_durations))
    counts = np.zeros(times.shape[0], dtype=int)
    log_likelihood = -encoding.rate_hz[None, :] * bin_durations[:, None]

    mark_times, mark_values = _marks_for_config(session, encoding.config)
    keep = (
        (mark_times >= ripple_event.start)
        & (mark_times < ripple_event.end)
        & np.all(np.isfinite(mark_values), axis=1)
    )
    if np.any(keep):
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        time_bins = np.searchsorted(edges, mark_times, side="right") - 1
        valid = (time_bins >= 0) & (time_bins < counts.shape[0])
        time_bins = time_bins[valid].astype(int)
        mark_values = mark_values[valid]
        log_rate = np.log(np.maximum(encoding.rate_hz, np.finfo(float).tiny))
        mark_log_likelihood = encoding.log_mark_likelihood(mark_values)
        for local_index, time_bin in enumerate(time_bins):
            log_likelihood[time_bin] += log_rate + mark_log_likelihood[local_index]
        np.add.at(counts, time_bins, 1)
    log_likelihood += (counts * np.log(bin_durations) - gammaln(counts + 1))[:, None]

    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=counts[:, None],
        times=times,
        dt=dt,
        cell_ids=np.array([0], dtype=int),
        n_spikes=int(counts.sum()),
    )


def _training_marks(
    session: ReplaySession,
    encoding_config: EncodingConfig,
    clusterless_config: ClusterlessMarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    mark_times, mark_values = _all_event_marks(session)
    if _use_excitatory_marks(clusterless_config, encoding_config) and marks.cell_ids is not None and session.excitatory_neurons.size:
        keep = np.isin(marks.cell_ids.astype(int), session.excitatory_neurons.astype(int))
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
    return mark_times, mark_values


def _marks_for_config(session: ReplaySession, config: ClusterlessMarkConfig) -> tuple[np.ndarray, np.ndarray]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    mark_times, mark_values = _all_event_marks(session)
    encoding_config = EncodingConfig() if config.encoding is None else config.encoding
    if _use_excitatory_marks(config, encoding_config) and marks.cell_ids is not None and session.excitatory_neurons.size:
        keep = np.isin(marks.cell_ids.astype(int), session.excitatory_neurons.astype(int))
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
    return mark_times, mark_values


def _use_excitatory_marks(
    clusterless_config: ClusterlessMarkConfig,
    encoding_config: EncodingConfig,
) -> bool:
    return bool(clusterless_config.use_excitatory and encoding_config.use_excitatory)


def _all_event_marks(session: ReplaySession) -> tuple[np.ndarray, np.ndarray]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    return np.asarray(marks.times, dtype=float), np.asarray(marks.marks, dtype=float)
