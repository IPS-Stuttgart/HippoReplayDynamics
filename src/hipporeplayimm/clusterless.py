"""Clusterless marked-point-process emissions for replay state-space models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln, logsumexp
from scipy.spatial import cKDTree

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

_DIAGONAL_GAUSSIAN = "diagonal-gaussian"
_LOCAL_KDE = "local-kde"
_MARK_LIKELIHOOD_ALIASES = {
    "diag": _DIAGONAL_GAUSSIAN,
    "diag-gaussian": _DIAGONAL_GAUSSIAN,
    "diagonal": _DIAGONAL_GAUSSIAN,
    "diagonal-gaussian": _DIAGONAL_GAUSSIAN,
    "gaussian": _DIAGONAL_GAUSSIAN,
    "kde": _LOCAL_KDE,
    "local-kde": _LOCAL_KDE,
    "local_kde": _LOCAL_KDE,
}


@dataclass(frozen=True)
class ClusterlessMarkConfig:
    """Configuration for clusterless marked-point-process encoding."""

    encoding: EncodingConfig | None = None
    mark_smoothing_sigma_bins: float = 1.0
    mark_prior_count: float = 1.0
    mark_variance_floor: float = 1.0
    rate_floor_hz: float = 1e-4
    use_excitatory: bool = True
    mark_likelihood: str = _LOCAL_KDE
    mark_kde_bandwidth: float | None = None
    mark_kde_spatial_sigma_bins: float | None = None
    mark_kde_max_neighbors: int = 256
    group_marks_by_tetrode: bool = True


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
    mark_likelihood: str
    mark_kde_values: np.ndarray | None = None
    mark_kde_neighbor_indices: np.ndarray | None = None
    mark_kde_log_weights: np.ndarray | None = None
    mark_kde_variance: np.ndarray | None = None
    mark_kde_group_ids: np.ndarray | None = None
    mark_group_labels: np.ndarray | None = None
    mark_group_rate_hz: np.ndarray | None = None

    @property
    def n_bins(self) -> int:
        return int(self.bin_centers.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.mark_mean.shape[1])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (len(self.x_edges) - 1, len(self.y_edges) - 1)

    @property
    def n_mark_groups(self) -> int:
        if self.mark_group_labels is None:
            return 1
        return int(np.asarray(self.mark_group_labels).reshape(-1).shape[0])

    def log_mark_likelihood(self, marks: np.ndarray, group_ids: np.ndarray | None = None) -> np.ndarray:
        marks = self._coerce_marks(marks)
        group_ids = self._coerce_mark_group_ids(group_ids, marks.shape[0])
        if self.mark_likelihood == _LOCAL_KDE:
            return self._log_mark_likelihood_local_kde(marks, group_ids=group_ids)
        return self._log_mark_likelihood_diagonal_gaussian(marks)

    def _coerce_marks(self, marks: np.ndarray) -> np.ndarray:
        marks = np.asarray(marks, dtype=float)
        if marks.ndim == 1:
            marks = marks[None, :]
        if marks.ndim != 2:
            raise ValueError("marks must be one- or two-dimensional")
        if marks.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} mark features, got {marks.shape[1]}")
        return marks

    def _coerce_mark_group_ids(self, group_ids: np.ndarray | None, n_marks: int) -> np.ndarray | None:
        if group_ids is None:
            return None
        coerced = np.asarray(group_ids, dtype=int).reshape(-1)
        if coerced.shape[0] != n_marks:
            raise ValueError(f"Expected {n_marks} mark group ids, got {coerced.shape[0]}")
        return coerced

    def _log_mark_likelihood_diagonal_gaussian(self, marks: np.ndarray) -> np.ndarray:
        diff = marks[:, None, :] - self.mark_mean[None, :, :]
        log_norm = np.sum(np.log(2.0 * np.pi * self.mark_variance), axis=1)
        quad = np.sum(diff * diff / self.mark_variance[None, :, :], axis=2)
        return -0.5 * (quad + log_norm[None, :])

    def _log_mark_likelihood_local_kde(
        self,
        marks: np.ndarray,
        group_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if (
            self.mark_kde_values is None
            or self.mark_kde_neighbor_indices is None
            or self.mark_kde_log_weights is None
            or self.mark_kde_variance is None
        ):
            raise ValueError("Local mark KDE parameters are missing from the clusterless encoding.")
        kde_values = np.asarray(self.mark_kde_values, dtype=float)
        indices = np.asarray(self.mark_kde_neighbor_indices, dtype=int)
        log_weights = np.asarray(self.mark_kde_log_weights, dtype=float)
        variance = np.asarray(self.mark_kde_variance, dtype=float)
        if variance.shape[0] != self.n_features:
            raise ValueError("Local mark KDE bandwidth dimensionality does not match the mark features.")

        log_norm = float(np.sum(np.log(2.0 * np.pi * variance)))
        output = np.empty((marks.shape[0], self.n_bins), dtype=float)
        grouped_kde_ids = None if self.mark_kde_group_ids is None else np.asarray(self.mark_kde_group_ids, dtype=int)
        fallback = None
        for bin_index in range(self.n_bins):
            support = indices[bin_index]
            weights = log_weights[bin_index]
            valid = (support >= 0) & np.isfinite(weights)
            if not np.any(valid):
                if fallback is None:
                    fallback = self._log_mark_likelihood_diagonal_gaussian(marks)
                output[:, bin_index] = fallback[:, bin_index]
                continue
            support_indices = support[valid]
            support_values = kde_values[support_indices]
            support_weights = weights[valid]
            if group_ids is not None and grouped_kde_ids is not None:
                if fallback is None:
                    fallback = self._log_mark_likelihood_diagonal_gaussian(marks)
                support_groups = grouped_kde_ids[support_indices]
                for mark_index, group_id in enumerate(group_ids):
                    same_group = support_groups == group_id
                    if not np.any(same_group):
                        output[mark_index, bin_index] = fallback[mark_index, bin_index]
                        continue
                    diff = marks[mark_index, None, :] - support_values[same_group]
                    quad = np.sum(diff * diff / variance[None, :], axis=1)
                    output[mark_index, bin_index] = logsumexp(
                        support_weights[same_group] - 0.5 * (quad + log_norm)
                    )
                continue
            diff = marks[:, None, :] - support_values[None, :, :]
            quad = np.sum(diff * diff / variance[None, None, :], axis=2)
            output[:, bin_index] = logsumexp(support_weights[None, :] - 0.5 * (quad + log_norm), axis=1)
        return output


@dataclass
class ClusterlessStateSpaceReplayModel(StateSpaceReplayModel):
    """State-space replay model using clusterless marked-point-process emissions."""

    mode: str = "diffusion"
    config: StateSpaceDecoderConfig | None = None
    name: str | None = None
    mark_likelihood: str = _LOCAL_KDE

    def __post_init__(self) -> None:
        super().__post_init__()
        self.mark_likelihood = _normalize_mark_likelihood(self.mark_likelihood)
        if self.name is None or self.name.startswith("state-space-"):
            self.name = f"clusterless-state-space-{self.mode}"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray] | None = None,
    ) -> EventScore:
        score = super().score(emissions, bin_centers, candidate_indices=candidate_indices)
        metadata = getattr(emissions, "metadata", {}) or {}
        score.model_name = str(self.name)
        score.diagnostics["state_space_observation_model"] = "clusterless-marked-point-process"
        score.diagnostics["clusterless_mark_likelihood"] = str(metadata.get("clusterless_mark_likelihood", self.mark_likelihood))
        if "clusterless_mark_kde_bandwidth" in metadata:
            score.diagnostics["clusterless_mark_kde_bandwidth"] = metadata["clusterless_mark_kde_bandwidth"]
        if "clusterless_mark_kde_max_neighbors" in metadata:
            score.diagnostics["clusterless_mark_kde_max_neighbors"] = metadata["clusterless_mark_kde_max_neighbors"]
        if "clusterless_group_marks_by_tetrode" in metadata:
            score.diagnostics["clusterless_group_marks_by_tetrode"] = metadata["clusterless_group_marks_by_tetrode"]
        if "clusterless_mark_group_count" in metadata:
            score.diagnostics["clusterless_mark_group_count"] = metadata["clusterless_mark_group_count"]
        if "clusterless_mark_groups" in metadata:
            score.diagnostics["clusterless_mark_groups"] = metadata["clusterless_mark_groups"]
        return score


def fit_clusterless_mark_encoding(
    session: ReplaySession,
    config: ClusterlessMarkConfig | None = None,
) -> ClusterlessMarkEncoding:
    """Fit clusterless marked-point-process parameters from behavioral run periods."""

    config = ClusterlessMarkConfig() if config is None else config
    mark_likelihood = _normalize_mark_likelihood(config.mark_likelihood)
    _validate_mark_config(config)
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

    mark_times, mark_values, mark_group_ids = _training_marks(session, encoding_config, config)
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
    mark_group_ids = mark_group_ids[keep]
    mark_bins = mark_bins[keep].astype(int)
    finite_rows = np.all(np.isfinite(mark_values), axis=1)
    mark_values = mark_values[finite_rows]
    mark_group_ids = mark_group_ids[finite_rows].astype(int)
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

    mark_group_labels, smooth_group_count = _smooth_mark_group_counts(
        mark_bins,
        mark_group_ids,
        grid_shape,
        config,
    )

    kde_values = None
    kde_neighbor_indices = None
    kde_log_weights = None
    kde_variance = None
    kde_group_ids = None
    if mark_likelihood == _LOCAL_KDE:
        kde_neighbor_indices, kde_log_weights = _local_kde_support(
            mark_bins,
            grid_shape,
            config,
        )
        kde_values = np.asarray(mark_values, dtype=float).copy()
        kde_variance = _mark_kde_variance(mark_values, config)
        if config.group_marks_by_tetrode:
            kde_group_ids = np.asarray(mark_group_ids, dtype=int).copy()

    occupancy_denominator = np.maximum(smooth_occupancy, encoding_config.min_occupancy_s)
    group_rate_floor_hz = config.rate_floor_hz / max(int(mark_group_labels.shape[0]), 1)
    group_rate_hz = np.maximum(
        smooth_group_count / occupancy_denominator[None, :],
        group_rate_floor_hz,
    )
    rate_hz = np.maximum(np.sum(group_rate_hz, axis=0), config.rate_floor_hz)
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
        mark_likelihood=mark_likelihood,
        mark_kde_values=kde_values,
        mark_kde_neighbor_indices=kde_neighbor_indices,
        mark_kde_log_weights=kde_log_weights,
        mark_kde_variance=kde_variance,
        mark_kde_group_ids=kde_group_ids,
        mark_group_labels=mark_group_labels,
        mark_group_rate_hz=group_rate_hz,
    )


def build_clusterless_mark_emissions(
    session: ReplaySession,
    encoding: ClusterlessMarkEncoding,
    ripple: RippleEvent | int,
    config: EmissionConfig | None = None,
) -> LogEmissionTensor:
    """Build marked-point-process log emissions for one ripple."""

    config = EmissionConfig() if config is None else config
    if config.spike_rate_scale <= 0.0:
        raise ValueError("spike_rate_scale must be positive")
    marks = session.spike_marks
    if marks is None or marks.n_features == 0:
        raise ValueError("Session does not contain spike marks for clusterless emission scoring.")
    ripple_event = session.ripple(ripple) if isinstance(ripple, int) else ripple
    edges = _time_bin_edges(ripple_event.start, ripple_event.end, config.time_bin_s)
    bin_durations = np.diff(edges)
    times = edges[:-1] + 0.5 * bin_durations
    dt = float(np.median(bin_durations))
    counts = np.zeros(times.shape[0], dtype=int)
    scaled_rate_hz = np.maximum(
        encoding.rate_hz * float(config.spike_rate_scale),
        np.finfo(float).tiny,
    )
    scaled_group_rate_hz, group_labels = _scaled_group_rate_hz(
        encoding,
        float(config.spike_rate_scale),
    )
    log_likelihood = -np.sum(scaled_group_rate_hz, axis=0)[None, :] * bin_durations[:, None]

    mark_times, mark_values, mark_group_ids = _marks_for_config(session, encoding.config)
    keep = (
        (mark_times >= ripple_event.start)
        & (mark_times < ripple_event.end)
        & np.all(np.isfinite(mark_values), axis=1)
    )
    if np.any(keep):
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        mark_group_ids = mark_group_ids[keep]
        time_bins = np.searchsorted(edges, mark_times, side="right") - 1
        valid = (time_bins >= 0) & (time_bins < counts.shape[0])
        time_bins = time_bins[valid].astype(int)
        mark_values = mark_values[valid]
        mark_group_ids = mark_group_ids[valid].astype(int)
        log_rate = _log_rate_by_mark_group(
            scaled_group_rate_hz,
            group_labels,
            mark_group_ids,
            fallback_rate_hz=scaled_rate_hz,
        )
        mark_log_likelihood = encoding.log_mark_likelihood(mark_values, group_ids=mark_group_ids)
        for local_index, time_bin in enumerate(time_bins):
            log_likelihood[time_bin] += log_rate[local_index] + mark_log_likelihood[local_index]
        np.add.at(counts, time_bins, 1)
    log_likelihood += (counts * np.log(bin_durations) - gammaln(counts + 1))[:, None]

    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=counts[:, None],
        times=times,
        dt=dt,
        cell_ids=np.array([0], dtype=int),
        n_spikes=int(counts.sum()),
    )
    emissions.metadata = {
        "clusterless_mark_likelihood": encoding.mark_likelihood,
        "clusterless_mark_kde_bandwidth": _format_float_array(encoding.mark_kde_variance),
        "clusterless_mark_kde_max_neighbors": _kde_neighbor_count(encoding),
        "clusterless_group_marks_by_tetrode": bool(encoding.config.group_marks_by_tetrode),
        "clusterless_mark_group_count": int(encoding.n_mark_groups),
        "clusterless_mark_groups": _format_int_array(encoding.mark_group_labels),
    }
    return emissions


def _validate_mark_config(config: ClusterlessMarkConfig) -> None:
    _normalize_mark_likelihood(config.mark_likelihood)
    if config.mark_smoothing_sigma_bins < 0.0:
        raise ValueError("mark_smoothing_sigma_bins must be nonnegative")
    if config.mark_prior_count < 0.0:
        raise ValueError("mark_prior_count must be nonnegative")
    if config.mark_variance_floor <= 0.0:
        raise ValueError("mark_variance_floor must be positive")
    if config.rate_floor_hz <= 0.0:
        raise ValueError("rate_floor_hz must be positive")
    if config.mark_kde_bandwidth is not None and config.mark_kde_bandwidth <= 0.0:
        raise ValueError("mark_kde_bandwidth must be positive when provided")
    if config.mark_kde_spatial_sigma_bins is not None and config.mark_kde_spatial_sigma_bins < 0.0:
        raise ValueError("mark_kde_spatial_sigma_bins must be nonnegative when provided")
    if int(config.mark_kde_max_neighbors) < 1:
        raise ValueError("mark_kde_max_neighbors must be at least one")


def _normalize_mark_likelihood(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    if key not in _MARK_LIKELIHOOD_ALIASES:
        allowed = ", ".join(sorted({_DIAGONAL_GAUSSIAN, _LOCAL_KDE}))
        raise ValueError(f"Unknown clusterless mark likelihood {value!r}; expected one of: {allowed}")
    return _MARK_LIKELIHOOD_ALIASES[key]


def _local_kde_support(
    mark_bins: np.ndarray,
    grid_shape: tuple[int, int],
    config: ClusterlessMarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mark_bins = np.asarray(mark_bins, dtype=int)
    n_marks = int(mark_bins.shape[0])
    n_bins = int(grid_shape[0] * grid_shape[1])
    support_size = min(int(config.mark_kde_max_neighbors), n_marks)
    mark_coords = np.column_stack(np.unravel_index(mark_bins, grid_shape)).astype(float)
    bin_coords = np.column_stack(np.unravel_index(np.arange(n_bins), grid_shape)).astype(float)
    distances, indices = cKDTree(mark_coords).query(bin_coords, k=support_size)
    if support_size == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    spatial_sigma = config.mark_kde_spatial_sigma_bins
    if spatial_sigma is None:
        spatial_sigma = config.mark_smoothing_sigma_bins
    spatial_sigma = float(spatial_sigma)
    if spatial_sigma > 0.0:
        weights = np.exp(-0.5 * (distances / spatial_sigma) ** 2)
    else:
        weights = (distances <= 0.0).astype(float)

    prior = max(float(config.mark_prior_count), 0.0)
    if prior > 0.0:
        weights = weights + prior / support_size
    empty = np.sum(weights, axis=1) <= np.finfo(float).tiny
    if np.any(empty):
        weights[empty] = 1.0
    log_weights = np.log(weights) - np.log(np.sum(weights, axis=1, keepdims=True))
    return np.asarray(indices, dtype=int), np.asarray(log_weights, dtype=float)


def _mark_kde_variance(mark_values: np.ndarray, config: ClusterlessMarkConfig) -> np.ndarray:
    values = np.asarray(mark_values, dtype=float)
    if config.mark_kde_bandwidth is None:
        n_marks = max(int(values.shape[0]), 1)
        n_features = max(int(values.shape[1]), 1)
        scale = n_marks ** (-1.0 / (n_features + 4.0))
        bandwidth = np.sqrt(np.maximum(np.var(values, axis=0), config.mark_variance_floor)) * scale
    else:
        bandwidth = np.full(values.shape[1], float(config.mark_kde_bandwidth), dtype=float)
    return np.maximum(bandwidth * bandwidth, config.mark_variance_floor)


def _smooth_mark_group_counts(
    mark_bins: np.ndarray,
    mark_group_ids: np.ndarray,
    grid_shape: tuple[int, int],
    config: ClusterlessMarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = int(grid_shape[0] * grid_shape[1])
    if config.group_marks_by_tetrode:
        group_ids = np.asarray(mark_group_ids, dtype=int).reshape(-1)
    else:
        group_ids = np.zeros(np.asarray(mark_bins).shape[0], dtype=int)
    labels = np.unique(group_ids)
    if labels.size == 0:
        labels = np.array([0], dtype=int)
    raw_group_count = np.zeros((labels.shape[0], n_bins), dtype=float)
    for group_index, label in enumerate(labels):
        in_group = group_ids == label
        if np.any(in_group):
            np.add.at(raw_group_count[group_index], np.asarray(mark_bins, dtype=int)[in_group], 1.0)
    sigma = float(config.mark_smoothing_sigma_bins)
    if sigma <= 0.0:
        return labels.astype(int), raw_group_count
    smooth_group_count = np.vstack(
        [
            gaussian_filter(row.reshape(grid_shape), sigma=sigma, mode="constant").reshape(-1)
            for row in raw_group_count
        ]
    )
    return labels.astype(int), smooth_group_count


def _scaled_group_rate_hz(
    encoding: ClusterlessMarkEncoding,
    spike_rate_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if encoding.mark_group_rate_hz is None or encoding.mark_group_labels is None:
        return (
            np.maximum(encoding.rate_hz[None, :] * spike_rate_scale, np.finfo(float).tiny),
            np.array([0], dtype=int),
        )
    return (
        np.maximum(np.asarray(encoding.mark_group_rate_hz, dtype=float) * spike_rate_scale, np.finfo(float).tiny),
        np.asarray(encoding.mark_group_labels, dtype=int).reshape(-1),
    )


def _log_rate_by_mark_group(
    scaled_group_rate_hz: np.ndarray,
    group_labels: np.ndarray,
    mark_group_ids: np.ndarray,
    *,
    fallback_rate_hz: np.ndarray,
) -> np.ndarray:
    output = np.empty((int(mark_group_ids.shape[0]), int(fallback_rate_hz.shape[0])), dtype=float)
    for mark_index, group_id in enumerate(np.asarray(mark_group_ids, dtype=int).reshape(-1)):
        matches = np.flatnonzero(group_labels == group_id)
        rate = fallback_rate_hz if matches.size == 0 else scaled_group_rate_hz[int(matches[0])]
        output[mark_index] = np.log(np.maximum(rate, np.finfo(float).tiny))
    return output


def _format_float_array(value: np.ndarray | None) -> str:
    if value is None:
        return ""
    return ",".join(f"{float(x):.6g}" for x in np.asarray(value, dtype=float).reshape(-1))


def _format_int_array(value: np.ndarray | None) -> str:
    if value is None:
        return ""
    return ",".join(str(int(x)) for x in np.asarray(value, dtype=int).reshape(-1))


def _kde_neighbor_count(encoding: ClusterlessMarkEncoding) -> int:
    if encoding.mark_kde_neighbor_indices is None:
        return 0
    return int(encoding.mark_kde_neighbor_indices.shape[1])


def _training_marks(
    session: ReplaySession,
    encoding_config: EncodingConfig,
    clusterless_config: ClusterlessMarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    mark_times, mark_values, mark_group_ids = _all_event_marks(session, clusterless_config)
    if _use_excitatory_marks(clusterless_config, encoding_config) and marks.cell_ids is not None and session.excitatory_neurons.size:
        keep = np.isin(marks.cell_ids.astype(int), session.excitatory_neurons.astype(int))
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        mark_group_ids = mark_group_ids[keep]
    return mark_times, mark_values, mark_group_ids


def _marks_for_config(session: ReplaySession, config: ClusterlessMarkConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    mark_times, mark_values, mark_group_ids = _all_event_marks(session, config)
    encoding_config = EncodingConfig() if config.encoding is None else config.encoding
    if _use_excitatory_marks(config, encoding_config) and marks.cell_ids is not None and session.excitatory_neurons.size:
        keep = np.isin(marks.cell_ids.astype(int), session.excitatory_neurons.astype(int))
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        mark_group_ids = mark_group_ids[keep]
    return mark_times, mark_values, mark_group_ids


def _use_excitatory_marks(
    clusterless_config: ClusterlessMarkConfig,
    encoding_config: EncodingConfig,
) -> bool:
    return bool(clusterless_config.use_excitatory and encoding_config.use_excitatory)


def _all_event_marks(
    session: ReplaySession,
    config: ClusterlessMarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    return (
        np.asarray(marks.times, dtype=float),
        np.asarray(marks.marks, dtype=float),
        _spike_mark_group_ids(session, config),
    )


def _spike_mark_group_ids(session: ReplaySession, config: ClusterlessMarkConfig) -> np.ndarray:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    if not config.group_marks_by_tetrode or marks.cell_ids is None:
        return np.zeros(marks.n_spikes, dtype=int)
    cell_to_tetrode = _cell_to_tetrode_map(session.tetrode_cell_ids)
    if not cell_to_tetrode:
        return np.zeros(marks.n_spikes, dtype=int)
    return np.asarray([cell_to_tetrode.get(int(cell_id), -1) for cell_id in marks.cell_ids], dtype=int)


def _cell_to_tetrode_map(tetrode_cell_ids: np.ndarray) -> dict[int, int]:
    values = np.asarray(tetrode_cell_ids)
    if values.ndim != 2 or values.shape[1] < 2:
        return {}
    mapping: dict[int, int] = {}
    for row in values:
        if np.all(np.isfinite(row[:2])):
            mapping[int(row[1])] = int(row[0])
    return mapping
