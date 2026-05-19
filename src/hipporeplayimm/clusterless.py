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
    group_marks_by_tetrode: bool = True
    mark_likelihood: str = _LOCAL_KDE
    mark_kde_bandwidth: float | None = None
    mark_kde_spatial_sigma_bins: float | None = None
    mark_kde_max_neighbors: int = 256


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

    @property
    def n_bins(self) -> int:
        return int(self.bin_centers.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.mark_mean.shape[1])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (len(self.x_edges) - 1, len(self.y_edges) - 1)

    def log_mark_likelihood(self, marks: np.ndarray, mark_group_ids: np.ndarray | None = None) -> np.ndarray:
        marks = self._coerce_marks(marks)
        mark_group_ids = self._coerce_mark_group_ids(mark_group_ids, marks.shape[0])
        if self.mark_likelihood == _LOCAL_KDE:
            return self._log_mark_likelihood_local_kde(marks, mark_group_ids=mark_group_ids)
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

    def _coerce_mark_group_ids(self, mark_group_ids: np.ndarray | None, n_marks: int) -> np.ndarray | None:
        if mark_group_ids is None:
            return None
        groups = np.asarray(mark_group_ids, dtype=int).reshape(-1)
        if groups.shape[0] != n_marks:
            raise ValueError(f"Expected {n_marks} mark group ids, got {groups.shape[0]}")
        return groups

    def _log_mark_likelihood_diagonal_gaussian(self, marks: np.ndarray) -> np.ndarray:
        diff = marks[:, None, :] - self.mark_mean[None, :, :]
        log_norm = np.sum(np.log(2.0 * np.pi * self.mark_variance), axis=1)
        quad = np.sum(diff * diff / self.mark_variance[None, :, :], axis=2)
        return -0.5 * (quad + log_norm[None, :])

    def _log_mark_likelihood_local_kde(
        self,
        marks: np.ndarray,
        mark_group_ids: np.ndarray | None = None,
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
        train_group_ids = None
        if self.mark_kde_group_ids is not None:
            train_group_ids = np.asarray(self.mark_kde_group_ids, dtype=int).reshape(-1)
            if train_group_ids.shape[0] != kde_values.shape[0]:
                raise ValueError("Local mark KDE group ids do not match the KDE support values.")

        log_norm = float(np.sum(np.log(2.0 * np.pi * variance)))
        output = np.empty((marks.shape[0], self.n_bins), dtype=float)
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
            valid_support = support[valid]
            valid_weights = weights[valid]
            if train_group_ids is None or mark_group_ids is None:
                output[:, bin_index] = self._evaluate_local_kde(
                    marks,
                    kde_values[valid_support],
                    valid_weights,
                    variance,
                    log_norm,
                )
                continue

            if fallback is None:
                fallback = self._log_mark_likelihood_diagonal_gaussian(marks)
            output[:, bin_index] = fallback[:, bin_index]
            support_group_ids = train_group_ids[valid_support]
            for group_id in np.unique(mark_group_ids):
                same_group = support_group_ids == group_id
                if not np.any(same_group):
                    continue
                mark_rows = mark_group_ids == group_id
                output[mark_rows, bin_index] = self._evaluate_local_kde(
                    marks[mark_rows],
                    kde_values[valid_support[same_group]],
                    valid_weights[same_group],
                    variance,
                    log_norm,
                )
        return output

    @staticmethod
    def _evaluate_local_kde(
        marks: np.ndarray,
        support_values: np.ndarray,
        log_weights: np.ndarray,
        variance: np.ndarray,
        log_norm: float,
    ) -> np.ndarray:
        log_weights = np.asarray(log_weights, dtype=float).reshape(-1)
        support_values = np.asarray(support_values, dtype=float)
        if support_values.shape[0] != log_weights.shape[0]:
            raise ValueError("Local mark KDE support and weights have inconsistent lengths.")
        normalized_log_weights = log_weights - logsumexp(log_weights)
        diff = marks[:, None, :] - support_values[None, :, :]
        quad = np.sum(diff * diff / variance[None, None, :], axis=2)
        return logsumexp(normalized_log_weights[None, :] - 0.5 * (quad + log_norm), axis=1)


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
        if "clusterless_mark_group_count" in metadata:
            score.diagnostics["clusterless_mark_group_count"] = metadata["clusterless_mark_group_count"]
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

    mark_times, mark_values, mark_groups = _training_marks(session, encoding_config, config)
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
    if mark_groups is not None:
        mark_groups = mark_groups[keep]
    finite_rows = np.all(np.isfinite(mark_values), axis=1)
    mark_values = mark_values[finite_rows]
    mark_bins = mark_bins[finite_rows]
    if mark_groups is not None:
        mark_groups = mark_groups[finite_rows]
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

    kde_values = None
    kde_neighbor_indices = None
    kde_log_weights = None
    kde_variance = None
    kde_group_ids = None
    if mark_likelihood == _LOCAL_KDE:
        kde_group_ids = None if mark_groups is None else np.asarray(mark_groups, dtype=int).copy()
        kde_neighbor_indices, kde_log_weights = _local_kde_support(
            mark_bins,
            grid_shape,
            config,
            mark_group_ids=kde_group_ids,
        )
        kde_values = np.asarray(mark_values, dtype=float).copy()
        kde_variance = _mark_kde_variance(mark_values, config)

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
        mark_likelihood=mark_likelihood,
        mark_kde_values=kde_values,
        mark_kde_neighbor_indices=kde_neighbor_indices,
        mark_kde_log_weights=kde_log_weights,
        mark_kde_variance=kde_variance,
        mark_kde_group_ids=kde_group_ids,
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
    log_likelihood = -scaled_rate_hz[None, :] * bin_durations[:, None]

    mark_times, mark_values, mark_groups = _marks_for_config(session, encoding.config)
    keep = (
        (mark_times >= ripple_event.start)
        & (mark_times < ripple_event.end)
        & np.all(np.isfinite(mark_values), axis=1)
    )
    if np.any(keep):
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        if mark_groups is not None:
            mark_groups = mark_groups[keep]
        time_bins = np.searchsorted(edges, mark_times, side="right") - 1
        valid = (time_bins >= 0) & (time_bins < counts.shape[0])
        time_bins = time_bins[valid].astype(int)
        mark_values = mark_values[valid]
        if mark_groups is not None:
            mark_groups = mark_groups[valid]
        log_rate = np.log(scaled_rate_hz)
        mark_log_likelihood = encoding.log_mark_likelihood(mark_values, mark_group_ids=mark_groups)
        for local_index, time_bin in enumerate(time_bins):
            log_likelihood[time_bin] += log_rate + mark_log_likelihood[local_index]
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
        "clusterless_mark_group_count": _kde_group_count(encoding),
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
    mark_group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mark_bins = np.asarray(mark_bins, dtype=int)
    n_marks = int(mark_bins.shape[0])
    n_bins = int(grid_shape[0] * grid_shape[1])
    mark_indices = np.arange(n_marks, dtype=int)
    if mark_group_ids is None:
        return _local_kde_support_for_indices(mark_bins, grid_shape, config, mark_indices)

    group_ids = np.asarray(mark_group_ids, dtype=int).reshape(-1)
    if group_ids.shape[0] != n_marks:
        raise ValueError("mark_group_ids must have one entry per training mark")
    chunks = []
    for group_id in np.unique(group_ids):
        group_indices = mark_indices[group_ids == group_id]
        if group_indices.size:
            chunks.append(_local_kde_support_for_indices(mark_bins, grid_shape, config, group_indices))
    if not chunks:
        return np.full((n_bins, 0), -1, dtype=int), np.full((n_bins, 0), -np.inf, dtype=float)
    total_support = int(sum(indices.shape[1] for indices, _ in chunks))
    indices = np.full((n_bins, total_support), -1, dtype=int)
    log_weights = np.full((n_bins, total_support), -np.inf, dtype=float)
    offset = 0
    for chunk_indices, chunk_log_weights in chunks:
        width = int(chunk_indices.shape[1])
        indices[:, offset : offset + width] = chunk_indices
        log_weights[:, offset : offset + width] = chunk_log_weights
        offset += width
    return indices, log_weights


def _local_kde_support_for_indices(
    mark_bins: np.ndarray,
    grid_shape: tuple[int, int],
    config: ClusterlessMarkConfig,
    mark_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mark_indices = np.asarray(mark_indices, dtype=int).reshape(-1)
    n_bins = int(grid_shape[0] * grid_shape[1])
    support_size = min(int(config.mark_kde_max_neighbors), int(mark_indices.shape[0]))
    if support_size < 1:
        return np.full((n_bins, 0), -1, dtype=int), np.full((n_bins, 0), -np.inf, dtype=float)
    mark_coords = np.column_stack(np.unravel_index(mark_bins[mark_indices], grid_shape)).astype(float)
    bin_coords = np.column_stack(np.unravel_index(np.arange(n_bins), grid_shape)).astype(float)
    distances, indices = cKDTree(mark_coords).query(bin_coords, k=support_size)
    if support_size == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    global_indices = mark_indices[np.asarray(indices, dtype=int)]
    log_weights = _local_kde_spatial_log_weights(distances, support_size, config)
    return np.asarray(global_indices, dtype=int), np.asarray(log_weights, dtype=float)


def _local_kde_spatial_log_weights(
    distances: np.ndarray,
    support_size: int,
    config: ClusterlessMarkConfig,
) -> np.ndarray:

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
    return np.log(weights) - np.log(np.sum(weights, axis=1, keepdims=True))


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


def _format_float_array(value: np.ndarray | None) -> str:
    if value is None:
        return ""
    return ",".join(f"{float(x):.6g}" for x in np.asarray(value, dtype=float).reshape(-1))


def _kde_neighbor_count(encoding: ClusterlessMarkEncoding) -> int:
    if encoding.mark_kde_neighbor_indices is None:
        return 0
    return int(encoding.mark_kde_neighbor_indices.shape[1])


def _kde_group_count(encoding: ClusterlessMarkEncoding) -> int:
    if encoding.mark_kde_group_ids is None:
        return 0
    return int(np.unique(np.asarray(encoding.mark_kde_group_ids, dtype=int)).shape[0])


def _training_marks(
    session: ReplaySession,
    encoding_config: EncodingConfig,
    clusterless_config: ClusterlessMarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    mark_times, mark_values, mark_groups = _all_event_marks(
        session,
        group_marks=clusterless_config.group_marks_by_tetrode,
    )
    if _use_excitatory_marks(clusterless_config, encoding_config) and marks.cell_ids is not None and session.excitatory_neurons.size:
        keep = np.isin(marks.cell_ids.astype(int), session.excitatory_neurons.astype(int))
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        if mark_groups is not None:
            mark_groups = mark_groups[keep]
    return mark_times, mark_values, mark_groups


def _marks_for_config(session: ReplaySession, config: ClusterlessMarkConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    mark_times, mark_values, mark_groups = _all_event_marks(
        session,
        group_marks=config.group_marks_by_tetrode,
    )
    encoding_config = EncodingConfig() if config.encoding is None else config.encoding
    if _use_excitatory_marks(config, encoding_config) and marks.cell_ids is not None and session.excitatory_neurons.size:
        keep = np.isin(marks.cell_ids.astype(int), session.excitatory_neurons.astype(int))
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        if mark_groups is not None:
            mark_groups = mark_groups[keep]
    return mark_times, mark_values, mark_groups


def _use_excitatory_marks(
    clusterless_config: ClusterlessMarkConfig,
    encoding_config: EncodingConfig,
) -> bool:
    return bool(clusterless_config.use_excitatory and encoding_config.use_excitatory)


def _all_event_marks(session: ReplaySession, *, group_marks: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    groups = _mark_group_ids(session) if group_marks else None
    return np.asarray(marks.times, dtype=float), np.asarray(marks.marks, dtype=float), groups


def _mark_group_ids(session: ReplaySession) -> np.ndarray | None:
    marks = session.spike_marks
    if marks is None or marks.cell_ids is None:
        return None
    cell_ids = np.asarray(marks.cell_ids, dtype=int).reshape(-1)
    if cell_ids.shape[0] != marks.n_spikes:
        return None
    cell_to_tetrode = _cell_to_tetrode_lookup(session.tetrode_cell_ids, cell_ids)
    if not cell_to_tetrode:
        return cell_ids.copy()
    return np.asarray([cell_to_tetrode.get(int(cell_id), int(cell_id)) for cell_id in cell_ids], dtype=int)


def _cell_to_tetrode_lookup(tetrode_cell_ids: np.ndarray, observed_cell_ids: np.ndarray) -> dict[int, int]:
    arr = np.asarray(tetrode_cell_ids)
    if arr.size == 0:
        return {}
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if arr.size < 2:
            return {}
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return {}
    if arr.shape[1] < 2 and arr.shape[0] >= 2:
        arr = arr.T
    if arr.shape[1] < 2:
        return {}
    arr = np.asarray(arr[:, :2], dtype=int)
    observed = {int(cell_id) for cell_id in np.asarray(observed_cell_ids, dtype=int).reshape(-1)}
    lookup = _lookup_from_candidate_columns(arr[:, 1], arr[:, 0], observed)
    if lookup:
        return lookup
    return _lookup_from_candidate_columns(arr[:, 0], arr[:, 1], observed)


def _lookup_from_candidate_columns(cell_column: np.ndarray, tetrode_column: np.ndarray, observed: set[int]) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for cell_id, tetrode_id in zip(cell_column, tetrode_column, strict=False):
        cell = int(cell_id)
        if cell in observed and cell not in lookup:
            lookup[cell] = int(tetrode_id)
    return lookup
