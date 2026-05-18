"""Clusterless marked-point-process emissions for replay state-space models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln, logsumexp

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
_GROUPED_GAUSSIAN_MIXTURE = "grouped-gaussian-mixture"
_MARK_LIKELIHOODS = {_DIAGONAL_GAUSSIAN, _GROUPED_GAUSSIAN_MIXTURE}
_MARK_GROUPINGS = {"all", "cell", "tetrode"}


@dataclass(frozen=True)
class ClusterlessMarkConfig:
    """Configuration for clusterless marked-point-process encoding.

    The default reproduces the historical pooled diagonal-Gaussian mark model.
    Set ``mark_likelihood='grouped-gaussian-mixture'`` and
    ``mark_group_by='tetrode'`` to score independent tetrode-specific marked
    point processes with full-covariance local/global Gaussian mixtures.
    """

    encoding: EncodingConfig | None = None
    mark_smoothing_sigma_bins: float = 1.0
    mark_prior_count: float = 1.0
    mark_variance_floor: float = 1.0
    rate_floor_hz: float = 1e-4
    use_excitatory: bool = True
    mark_likelihood: str = _DIAGONAL_GAUSSIAN
    mark_group_by: str = "all"


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
    group_ids: np.ndarray | None = None
    group_rate_hz: np.ndarray | None = None
    group_effective_spike_count: np.ndarray | None = None
    group_mark_mean: np.ndarray | None = None
    group_mark_covariance: np.ndarray | None = None
    group_mark_precision: np.ndarray | None = None
    group_mark_log_det: np.ndarray | None = None
    group_global_mean: np.ndarray | None = None
    group_global_covariance: np.ndarray | None = None
    group_global_precision: np.ndarray | None = None
    group_global_log_det: np.ndarray | None = None
    group_mixture_weight: np.ndarray | None = None

    @property
    def n_bins(self) -> int:
        return int(self.bin_centers.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.mark_mean.shape[1])

    @property
    def n_mark_groups(self) -> int:
        return 1 if self.group_ids is None else int(self.group_ids.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (len(self.x_edges) - 1, len(self.y_edges) - 1)

    @property
    def uses_grouped_mark_model(self) -> bool:
        return self.config.mark_likelihood == _GROUPED_GAUSSIAN_MIXTURE

    def log_mark_likelihood(
        self,
        marks: np.ndarray,
        group_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        marks = np.asarray(marks, dtype=float)
        if marks.ndim == 1:
            marks = marks[None, :]
        if marks.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} mark features, got {marks.shape[1]}")
        if not self.uses_grouped_mark_model:
            return self._log_diagonal_mark_likelihood(marks)
        if group_ids is None:
            raise ValueError("group_ids are required for grouped clusterless mark likelihoods")
        return self._log_grouped_gaussian_mixture(marks, np.asarray(group_ids, dtype=int))

    def _log_diagonal_mark_likelihood(self, marks: np.ndarray) -> np.ndarray:
        diff = marks[:, None, :] - self.mark_mean[None, :, :]
        log_norm = np.sum(np.log(2.0 * np.pi * self.mark_variance), axis=1)
        quad = np.sum(diff * diff / self.mark_variance[None, :, :], axis=2)
        return -0.5 * (quad + log_norm[None, :])

    def _log_grouped_gaussian_mixture(self, marks: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
        if group_ids.shape[0] != marks.shape[0]:
            raise ValueError("group_ids must contain one group per mark")
        required = (
            self.group_ids,
            self.group_mark_mean,
            self.group_mark_precision,
            self.group_mark_log_det,
            self.group_global_mean,
            self.group_global_precision,
            self.group_global_log_det,
            self.group_mixture_weight,
        )
        if any(value is None for value in required):
            raise ValueError("Grouped clusterless mark parameters were not fitted")
        assert self.group_ids is not None
        assert self.group_mark_mean is not None
        assert self.group_mark_precision is not None
        assert self.group_mark_log_det is not None
        assert self.group_global_mean is not None
        assert self.group_global_precision is not None
        assert self.group_global_log_det is not None
        assert self.group_mixture_weight is not None

        out = np.empty((marks.shape[0], self.n_bins), dtype=float)
        tiny = np.finfo(float).tiny
        for mark_index, (mark, group_id) in enumerate(zip(marks, group_ids, strict=True)):
            group_matches = np.flatnonzero(self.group_ids == int(group_id))
            if group_matches.size == 0:
                out[mark_index] = self._log_diagonal_mark_likelihood(mark[None, :])[0]
                continue
            group_index = int(group_matches[0])
            local_log = _log_full_gaussian_by_bin(
                mark,
                self.group_mark_mean[group_index],
                self.group_mark_precision[group_index],
                self.group_mark_log_det[group_index],
            )
            global_log = _log_full_gaussian_global(
                mark,
                self.group_global_mean[group_index],
                self.group_global_precision[group_index],
                float(self.group_global_log_det[group_index]),
            )
            local_weight = np.clip(self.group_mixture_weight[group_index], 0.0, 1.0)
            components = np.vstack(
                [
                    np.log(np.maximum(local_weight, tiny)) + local_log,
                    np.log(np.maximum(1.0 - local_weight, tiny)) + global_log,
                ]
            )
            out[mark_index] = logsumexp(components, axis=0)
        return out


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
        score.diagnostics["clusterless_mark_likelihood"] = getattr(
            emissions,
            "clusterless_mark_likelihood",
            _DIAGONAL_GAUSSIAN,
        )
        score.diagnostics["clusterless_mark_group_by"] = getattr(
            emissions,
            "clusterless_mark_group_by",
            "all",
        )
        score.diagnostics["clusterless_mark_groups"] = int(
            getattr(emissions, "clusterless_mark_groups", 1)
        )
        return score


def fit_clusterless_mark_encoding(
    session: ReplaySession,
    config: ClusterlessMarkConfig | None = None,
) -> ClusterlessMarkEncoding:
    """Fit clusterless marked-point-process parameters from behavioral run periods."""

    config = ClusterlessMarkConfig() if config is None else config
    _validate_clusterless_config(config)
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
    mark_values = mark_values[keep]
    mark_group_ids = mark_group_ids[keep]
    mark_bins = mark_bins[keep].astype(int)
    finite_rows = np.all(np.isfinite(mark_values), axis=1)
    mark_values = mark_values[finite_rows]
    mark_group_ids = mark_group_ids[finite_rows]
    mark_bins = mark_bins[finite_rows]
    if mark_values.shape[0] == 0:
        raise ValueError("No finite run-period spike marks were available for clusterless encoding.")

    raw_count, mark_sum, mark_sq_sum = _accumulate_diagonal_mark_stats(
        mark_values,
        mark_bins,
        occupancy.shape[0],
    )
    smooth_occupancy = _smooth_flat(occupancy, grid_shape, encoding_config.smoothing_sigma_bins)
    smooth_count = _smooth_flat(raw_count, grid_shape, config.mark_smoothing_sigma_bins)
    smooth_sum = np.vstack(
        [_smooth_flat(row, grid_shape, config.mark_smoothing_sigma_bins) for row in mark_sum]
    )
    smooth_sq_sum = np.vstack(
        [_smooth_flat(row, grid_shape, config.mark_smoothing_sigma_bins) for row in mark_sq_sum]
    )

    global_mean = np.mean(mark_values, axis=0)
    global_variance = np.maximum(np.var(mark_values, axis=0), config.mark_variance_floor)
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
    grouped_params = _fit_grouped_mark_model(
        mark_values,
        mark_bins,
        mark_group_ids,
        smooth_occupancy,
        grid_shape,
        encoding_config,
        config,
    )
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
        **grouped_params,
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

    mark_times, mark_values, mark_group_ids = _marks_for_config(session, encoding.config)
    if encoding.uses_grouped_mark_model:
        tensor = _build_grouped_clusterless_emissions(
            encoding,
            mark_times,
            mark_values,
            mark_group_ids,
            ripple_event,
            edges,
            bin_durations,
            times,
            dt,
            float(config.spike_rate_scale),
        )
    else:
        tensor = _build_diagonal_clusterless_emissions(
            encoding,
            mark_times,
            mark_values,
            ripple_event,
            edges,
            bin_durations,
            times,
            dt,
            float(config.spike_rate_scale),
        )
    tensor.clusterless_mark_likelihood = encoding.config.mark_likelihood
    tensor.clusterless_mark_group_by = encoding.config.mark_group_by
    tensor.clusterless_mark_groups = encoding.n_mark_groups
    return tensor


def _build_diagonal_clusterless_emissions(
    encoding: ClusterlessMarkEncoding,
    mark_times: np.ndarray,
    mark_values: np.ndarray,
    ripple_event: RippleEvent,
    edges: np.ndarray,
    bin_durations: np.ndarray,
    times: np.ndarray,
    dt: float,
    spike_rate_scale: float,
) -> LogEmissionTensor:
    counts = np.zeros(times.shape[0], dtype=int)
    scaled_rate_hz = np.maximum(
        encoding.rate_hz * spike_rate_scale,
        np.finfo(float).tiny,
    )
    log_likelihood = -scaled_rate_hz[None, :] * bin_durations[:, None]
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
        log_rate = np.log(scaled_rate_hz)
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


def _build_grouped_clusterless_emissions(
    encoding: ClusterlessMarkEncoding,
    mark_times: np.ndarray,
    mark_values: np.ndarray,
    mark_group_ids: np.ndarray,
    ripple_event: RippleEvent,
    edges: np.ndarray,
    bin_durations: np.ndarray,
    times: np.ndarray,
    dt: float,
    spike_rate_scale: float,
) -> LogEmissionTensor:
    if encoding.group_ids is None or encoding.group_rate_hz is None:
        raise ValueError("Grouped clusterless emission scoring requires fitted group rates")
    counts = np.zeros((times.shape[0], encoding.n_mark_groups), dtype=int)
    scaled_rate_hz = np.maximum(
        encoding.group_rate_hz * spike_rate_scale,
        np.finfo(float).tiny,
    )
    log_likelihood = -scaled_rate_hz.sum(axis=0)[None, :] * bin_durations[:, None]
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
        group_rows = np.searchsorted(encoding.group_ids, mark_group_ids)
        valid = (time_bins >= 0) & (time_bins < counts.shape[0])
        valid &= (group_rows >= 0) & (group_rows < encoding.group_ids.shape[0])
        valid[valid] &= encoding.group_ids[group_rows[valid]] == mark_group_ids[valid]
        time_bins = time_bins[valid].astype(int)
        group_rows = group_rows[valid].astype(int)
        mark_values = mark_values[valid]
        mark_group_ids = mark_group_ids[valid]
        mark_log_likelihood = encoding.log_mark_likelihood(mark_values, mark_group_ids)
        for local_index, (time_bin, group_row) in enumerate(zip(time_bins, group_rows, strict=True)):
            log_likelihood[time_bin] += np.log(scaled_rate_hz[group_row]) + mark_log_likelihood[local_index]
        np.add.at(counts, (time_bins, group_rows), 1)
    poisson_time_terms = counts * np.log(bin_durations[:, None]) - gammaln(counts + 1)
    log_likelihood += poisson_time_terms.sum(axis=1)[:, None]

    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=counts,
        times=times,
        dt=dt,
        cell_ids=encoding.group_ids.copy(),
        n_spikes=int(counts.sum()),
    )


def _fit_grouped_mark_model(
    mark_values: np.ndarray,
    mark_bins: np.ndarray,
    mark_group_ids: np.ndarray,
    smooth_occupancy: np.ndarray,
    grid_shape: tuple[int, int],
    encoding_config: EncodingConfig,
    config: ClusterlessMarkConfig,
) -> dict[str, np.ndarray | None]:
    group_ids = np.asarray(sorted(np.unique(mark_group_ids)), dtype=int)
    if group_ids.size == 0:
        group_ids = np.array([0], dtype=int)
    n_groups = int(group_ids.shape[0])
    n_features = int(mark_values.shape[1])
    n_bins = int(smooth_occupancy.shape[0])
    group_rate_hz = np.zeros((n_groups, n_bins), dtype=float)
    group_effective_spike_count = np.zeros((n_groups, n_bins), dtype=float)
    group_mark_mean = np.zeros((n_groups, n_bins, n_features), dtype=float)
    group_mark_covariance = np.zeros((n_groups, n_bins, n_features, n_features), dtype=float)
    group_mark_precision = np.zeros_like(group_mark_covariance)
    group_mark_log_det = np.zeros((n_groups, n_bins), dtype=float)
    group_global_mean = np.zeros((n_groups, n_features), dtype=float)
    group_global_covariance = np.zeros((n_groups, n_features, n_features), dtype=float)
    group_global_precision = np.zeros_like(group_global_covariance)
    group_global_log_det = np.zeros(n_groups, dtype=float)
    group_mixture_weight = np.zeros((n_groups, n_bins), dtype=float)
    occupancy_denominator = np.maximum(smooth_occupancy, encoding_config.min_occupancy_s)
    prior = max(float(config.mark_prior_count), 0.0)
    tiny = np.finfo(float).tiny

    for group_index, group_id in enumerate(group_ids):
        keep = mark_group_ids == group_id
        group_marks = mark_values[keep]
        group_bins = mark_bins[keep]
        raw_count, mark_sum, mark_outer_sum = _accumulate_full_mark_stats(
            group_marks,
            group_bins,
            n_bins,
        )
        smooth_count = _smooth_flat(raw_count, grid_shape, config.mark_smoothing_sigma_bins)
        smooth_sum = np.vstack(
            [_smooth_flat(row, grid_shape, config.mark_smoothing_sigma_bins) for row in mark_sum]
        )
        smooth_outer = np.empty((n_features, n_features, n_bins), dtype=float)
        for row in range(n_features):
            for col in range(n_features):
                smooth_outer[row, col] = _smooth_flat(
                    mark_outer_sum[row, col],
                    grid_shape,
                    config.mark_smoothing_sigma_bins,
                )

        global_mean = np.mean(group_marks, axis=0)
        centered = group_marks - global_mean[None, :]
        global_cov = centered.T @ centered / max(int(group_marks.shape[0]), 1)
        global_cov, global_precision, global_log_det = _regularize_covariances(
            global_cov[None, :, :],
            config.mark_variance_floor,
        )
        group_global_mean[group_index] = global_mean
        group_global_covariance[group_index] = global_cov[0]
        group_global_precision[group_index] = global_precision[0]
        group_global_log_det[group_index] = global_log_det[0]

        safe_count = np.maximum(smooth_count, tiny)
        mean = smooth_sum.T / safe_count[:, None]
        second_moment = np.moveaxis(smooth_outer, 2, 0) / safe_count[:, None, None]
        covariance = second_moment - mean[:, :, None] * mean[:, None, :]
        no_support = smooth_count <= tiny
        if np.any(no_support):
            mean[no_support] = global_mean
            covariance[no_support] = global_cov[0]
        covariance, precision, log_det = _regularize_covariances(
            covariance,
            config.mark_variance_floor,
        )

        if prior > 0.0:
            mixture_weight = smooth_count / (smooth_count + prior)
        else:
            mixture_weight = (smooth_count > tiny).astype(float)
        mixture_weight[no_support] = 0.0
        group_effective_spike_count[group_index] = smooth_count
        group_mark_mean[group_index] = mean
        group_mark_covariance[group_index] = covariance
        group_mark_precision[group_index] = precision
        group_mark_log_det[group_index] = log_det
        group_mixture_weight[group_index] = np.clip(mixture_weight, 0.0, 1.0)
        group_rate_hz[group_index] = np.maximum(
            smooth_count / occupancy_denominator,
            config.rate_floor_hz,
        )

    return {
        "group_ids": group_ids,
        "group_rate_hz": group_rate_hz,
        "group_effective_spike_count": group_effective_spike_count,
        "group_mark_mean": group_mark_mean,
        "group_mark_covariance": group_mark_covariance,
        "group_mark_precision": group_mark_precision,
        "group_mark_log_det": group_mark_log_det,
        "group_global_mean": group_global_mean,
        "group_global_covariance": group_global_covariance,
        "group_global_precision": group_global_precision,
        "group_global_log_det": group_global_log_det,
        "group_mixture_weight": group_mixture_weight,
    }


def _accumulate_diagonal_mark_stats(
    mark_values: np.ndarray,
    mark_bins: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_count = np.zeros(n_bins, dtype=float)
    mark_sum = np.zeros((mark_values.shape[1], n_bins), dtype=float)
    mark_sq_sum = np.zeros_like(mark_sum)
    np.add.at(raw_count, mark_bins, 1.0)
    for feature_index in range(mark_values.shape[1]):
        feature = mark_values[:, feature_index]
        np.add.at(mark_sum[feature_index], mark_bins, feature)
        np.add.at(mark_sq_sum[feature_index], mark_bins, feature * feature)
    return raw_count, mark_sum, mark_sq_sum


def _accumulate_full_mark_stats(
    mark_values: np.ndarray,
    mark_bins: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = int(mark_values.shape[1])
    raw_count = np.zeros(n_bins, dtype=float)
    mark_sum = np.zeros((n_features, n_bins), dtype=float)
    mark_outer_sum = np.zeros((n_features, n_features, n_bins), dtype=float)
    if mark_values.shape[0] == 0:
        return raw_count, mark_sum, mark_outer_sum
    np.add.at(raw_count, mark_bins, 1.0)
    for feature_index in range(n_features):
        feature = mark_values[:, feature_index]
        np.add.at(mark_sum[feature_index], mark_bins, feature)
    for row in range(n_features):
        for col in range(n_features):
            np.add.at(mark_outer_sum[row, col], mark_bins, mark_values[:, row] * mark_values[:, col])
    return raw_count, mark_sum, mark_outer_sum


def _smooth_flat(values: np.ndarray, grid_shape: tuple[int, int], sigma: float) -> np.ndarray:
    if float(sigma) <= 0.0:
        return np.asarray(values, dtype=float).copy()
    return gaussian_filter(
        np.asarray(values, dtype=float).reshape(grid_shape),
        sigma=float(sigma),
        mode="constant",
    ).reshape(-1)


def _regularize_covariances(
    covariance: np.ndarray,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariance = np.asarray(covariance, dtype=float)
    n_features = int(covariance.shape[-1])
    flat = covariance.reshape(-1, n_features, n_features)
    cov_out = np.empty_like(flat)
    precision = np.empty_like(flat)
    log_det = np.empty(flat.shape[0], dtype=float)
    floor = float(variance_floor)
    identity = np.eye(n_features)
    for index, cov in enumerate(flat):
        sym = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.maximum(eigvals, floor)
        regularized = (eigvecs * eigvals[None, :]) @ eigvecs.T
        regularized = 0.5 * (regularized + regularized.T)
        sign, value = np.linalg.slogdet(regularized)
        if sign <= 0.0 or not np.isfinite(value):
            regularized = regularized + floor * identity
            sign, value = np.linalg.slogdet(regularized)
        cov_out[index] = regularized
        precision[index] = np.linalg.inv(regularized)
        log_det[index] = float(value)
    return (
        cov_out.reshape(covariance.shape),
        precision.reshape(covariance.shape),
        log_det.reshape(covariance.shape[:-2]),
    )


def _log_full_gaussian_by_bin(
    mark: np.ndarray,
    mean_by_bin: np.ndarray,
    precision_by_bin: np.ndarray,
    log_det_by_bin: np.ndarray,
) -> np.ndarray:
    diff = mark[None, :] - mean_by_bin
    quad = np.einsum("bf,bfg,bg->b", diff, precision_by_bin, diff, optimize=True)
    n_features = int(mean_by_bin.shape[1])
    return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det_by_bin + quad)


def _log_full_gaussian_global(
    mark: np.ndarray,
    mean: np.ndarray,
    precision: np.ndarray,
    log_det: float,
) -> float:
    diff = mark - mean
    quad = float(diff @ precision @ diff)
    n_features = int(mean.shape[0])
    return float(-0.5 * (n_features * np.log(2.0 * np.pi) + log_det + quad))


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


def _marks_for_config(
    session: ReplaySession,
    config: ClusterlessMarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        _mark_group_ids(session, config),
    )


def _mark_group_ids(session: ReplaySession, config: ClusterlessMarkConfig) -> np.ndarray:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    grouping = str(config.mark_group_by)
    n_spikes = int(marks.n_spikes)
    if grouping == "all":
        return np.zeros(n_spikes, dtype=int)
    if marks.cell_ids is None:
        raise ValueError(f"mark_group_by={grouping!r} requires spike-mark cell IDs")
    cell_ids = np.asarray(marks.cell_ids, dtype=int)
    if grouping == "cell":
        return cell_ids.copy()
    if grouping == "tetrode":
        mapping = _cell_to_tetrode_map(session.tetrode_cell_ids)
        if not mapping:
            raise ValueError("mark_group_by='tetrode' requires non-empty tetrode_cell_ids")
        missing = sorted({int(cell_id) for cell_id in cell_ids if int(cell_id) not in mapping})
        if missing:
            preview = ", ".join(str(cell_id) for cell_id in missing[:5])
            raise ValueError(f"No tetrode ID is available for cell IDs: {preview}")
        return np.asarray([mapping[int(cell_id)] for cell_id in cell_ids], dtype=int)
    raise ValueError(f"Unknown mark_group_by={grouping!r}; expected one of {sorted(_MARK_GROUPINGS)}")


def _cell_to_tetrode_map(tetrode_cell_ids: np.ndarray) -> dict[int, int]:
    arr = np.asarray(tetrode_cell_ids)
    if arr.size == 0:
        return {}
    arr = np.asarray(np.squeeze(arr), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            return {}
        arr = arr.reshape(1, 2)
    if arr.ndim != 2:
        return {}
    if arr.shape[1] != 2 and arr.shape[0] == 2:
        arr = arr.T
    if arr.shape[1] < 2:
        return {}
    mapping: dict[int, int] = {}
    for row in arr:
        if np.all(np.isfinite(row[:2])):
            tetrode_id = int(row[0])
            cell_id = int(row[1])
            mapping[cell_id] = tetrode_id
    return mapping


def _validate_clusterless_config(config: ClusterlessMarkConfig) -> None:
    if config.mark_likelihood not in _MARK_LIKELIHOODS:
        raise ValueError(
            f"mark_likelihood must be one of {sorted(_MARK_LIKELIHOODS)}, "
            f"got {config.mark_likelihood!r}"
        )
    if config.mark_group_by not in _MARK_GROUPINGS:
        raise ValueError(
            f"mark_group_by must be one of {sorted(_MARK_GROUPINGS)}, got {config.mark_group_by!r}"
        )
    if config.mark_smoothing_sigma_bins < 0.0:
        raise ValueError("mark_smoothing_sigma_bins must be nonnegative")
    if config.mark_prior_count < 0.0:
        raise ValueError("mark_prior_count must be nonnegative")
    if config.mark_variance_floor <= 0.0:
        raise ValueError("mark_variance_floor must be positive")
    if config.rate_floor_hz <= 0.0:
        raise ValueError("rate_floor_hz must be positive")
