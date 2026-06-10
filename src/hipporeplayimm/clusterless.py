"""Clusterless marked-point-process emissions for replay state-space models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln, logsumexp
from scipy.spatial import cKDTree

from .data import ReplaySession, RippleEvent, _coerce_ripple_event
from .encoding import (
    EmissionConfig,
    EncodingConfig,
    LogEmissionTensor,
    _clean_position,
    _encoding_exclusion_intervals,
    _apply_likelihood_temperature,
    _validate_emission_calibration,
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
_MARK_GROUP_BY_ALIASES = {
    "": "none",
    "auto": "auto",
    "cell": "cell",
    "cells": "cell",
    "none": "none",
    "off": "none",
    "tetrode": "tetrode",
    "tetrodes": "tetrode",
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
    mark_group_by: str = "auto"


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
    group_ids: np.ndarray | None = None
    group_rate_hz: np.ndarray | None = None
    group_effective_spike_count: np.ndarray | None = None
    group_mark_mean: np.ndarray | None = None
    group_mark_variance: np.ndarray | None = None
    group_mark_kde_neighbor_indices: np.ndarray | None = None
    group_mark_kde_log_weights: np.ndarray | None = None
    group_mark_kde_variance: np.ndarray | None = None

    @property
    def n_bins(self) -> int:
        return int(self.bin_centers.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.mark_mean.shape[1])

    @property
    def n_mark_groups(self) -> int:
        return 0 if self.group_ids is None else int(self.group_ids.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (len(self.x_edges) - 1, len(self.y_edges) - 1)

    def log_mark_likelihood(self, marks: np.ndarray, group_ids: np.ndarray | None = None) -> np.ndarray:
        marks = self._coerce_marks(marks)
        group_indices = self._coerce_group_indices(group_ids, marks.shape[0])
        if self.mark_likelihood == _LOCAL_KDE:
            return self._log_mark_likelihood_local_kde(marks, group_indices)
        return self._log_mark_likelihood_diagonal_gaussian(marks, group_indices)

    def _coerce_marks(self, marks: np.ndarray) -> np.ndarray:
        marks = np.asarray(marks, dtype=float)
        if marks.ndim == 1:
            marks = marks[None, :]
        if marks.ndim != 2:
            raise ValueError("marks must be one- or two-dimensional")
        if marks.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} mark features, got {marks.shape[1]}")
        return marks

    def _coerce_group_indices(self, group_ids: np.ndarray | None, n_marks: int) -> np.ndarray | None:
        if group_ids is None or self.group_ids is None:
            return None
        raw_group_ids = np.asarray(group_ids)
        if raw_group_ids.ndim == 0:
            raw_group_ids = np.full(n_marks, raw_group_ids.item())
        raw_group_ids = raw_group_ids.reshape(-1)
        if raw_group_ids.shape[0] != n_marks:
            raise ValueError(f"Expected {n_marks} mark group IDs, got {raw_group_ids.shape[0]}")
        numeric_group_ids = raw_group_ids.astype(float, copy=False)
        if not np.all(np.isfinite(numeric_group_ids)):
            raise ValueError("mark group IDs must be finite")
        integer_valued = np.isclose(
            numeric_group_ids,
            np.rint(numeric_group_ids),
            rtol=0.0,
            atol=0.0,
        )
        if not np.all(integer_valued):
            raise ValueError("mark group IDs must be integer-valued")
        coerced = numeric_group_ids.astype(int)
        sorted_order = np.argsort(self.group_ids)
        sorted_groups = np.asarray(self.group_ids, dtype=int)[sorted_order]
        positions = np.searchsorted(sorted_groups, coerced)
        in_bounds = positions < sorted_groups.shape[0]
        matches = np.zeros(n_marks, dtype=bool)
        matches[in_bounds] = sorted_groups[positions[in_bounds]] == coerced[in_bounds]
        group_indices = np.full(n_marks, -1, dtype=int)
        group_indices[matches] = sorted_order[positions[matches]]
        return group_indices

    def _log_mark_likelihood_diagonal_gaussian(self, marks: np.ndarray, group_indices: np.ndarray | None = None) -> np.ndarray:
        if group_indices is not None and self.group_mark_mean is not None and self.group_mark_variance is not None:
            output = np.empty((marks.shape[0], self.n_bins), dtype=float)
            global_likelihood = None
            for row_index, group_index in enumerate(group_indices):
                if group_index < 0:
                    if global_likelihood is None:
                        global_likelihood = self._log_mark_likelihood_diagonal_gaussian(marks)
                    output[row_index] = global_likelihood[row_index]
                    continue
                mean = self.group_mark_mean[int(group_index)]
                variance = self.group_mark_variance[int(group_index)]
                diff = marks[row_index][None, :] - mean
                log_norm = np.sum(np.log(2.0 * np.pi * variance), axis=1)
                quad = np.sum(diff * diff / variance, axis=1)
                output[row_index] = -0.5 * (quad + log_norm)
            return output
        diff = marks[:, None, :] - self.mark_mean[None, :, :]
        log_norm = np.sum(np.log(2.0 * np.pi * self.mark_variance), axis=1)
        quad = np.sum(diff * diff / self.mark_variance[None, :, :], axis=2)
        return -0.5 * (quad + log_norm[None, :])

    def _log_mark_likelihood_local_kde(self, marks: np.ndarray, group_indices: np.ndarray | None = None) -> np.ndarray:
        if (
            self.mark_kde_values is None
            or self.mark_kde_neighbor_indices is None
            or self.mark_kde_log_weights is None
            or self.mark_kde_variance is None
        ):
            raise ValueError("Local mark KDE parameters are missing from the clusterless encoding.")
        if (
            group_indices is not None
            and self.group_mark_kde_neighbor_indices is not None
            and self.group_mark_kde_log_weights is not None
            and self.group_mark_kde_variance is not None
        ):
            return self._log_mark_likelihood_grouped_local_kde(marks, group_indices)

        kde_values = np.asarray(self.mark_kde_values, dtype=float)
        indices = np.asarray(self.mark_kde_neighbor_indices, dtype=int)
        log_weights = np.asarray(self.mark_kde_log_weights, dtype=float)
        variance = np.asarray(self.mark_kde_variance, dtype=float)
        if variance.shape[0] != self.n_features:
            raise ValueError("Local mark KDE bandwidth dimensionality does not match the mark features.")

        log_norm = float(np.sum(np.log(2.0 * np.pi * variance)))
        output = np.empty((marks.shape[0], self.n_bins), dtype=float)
        for bin_index in range(self.n_bins):
            support = indices[bin_index]
            weights = log_weights[bin_index]
            valid = (support >= 0) & np.isfinite(weights)
            if not np.any(valid):
                output[:, bin_index] = self._log_mark_likelihood_diagonal_gaussian(marks)[:, bin_index]
                continue
            support_values = kde_values[support[valid]]
            diff = marks[:, None, :] - support_values[None, :, :]
            quad = np.sum(diff * diff / variance[None, None, :], axis=2)
            output[:, bin_index] = logsumexp(weights[valid][None, :] - 0.5 * (quad + log_norm), axis=1)
        return output

    def _log_mark_likelihood_grouped_local_kde(self, marks: np.ndarray, group_indices: np.ndarray) -> np.ndarray:
        if (
            self.mark_kde_values is None
            or self.group_mark_kde_neighbor_indices is None
            or self.group_mark_kde_log_weights is None
            or self.group_mark_kde_variance is None
        ):
            raise ValueError("Grouped local mark KDE parameters are missing from the clusterless encoding.")
        kde_values = np.asarray(self.mark_kde_values, dtype=float)
        neighbor_indices = np.asarray(self.group_mark_kde_neighbor_indices, dtype=int)
        log_weights = np.asarray(self.group_mark_kde_log_weights, dtype=float)
        variances = np.asarray(self.group_mark_kde_variance, dtype=float)
        output = np.empty((marks.shape[0], self.n_bins), dtype=float)
        global_likelihood = None
        for row_index, group_index in enumerate(group_indices):
            if group_index < 0:
                if global_likelihood is None:
                    global_likelihood = self._log_mark_likelihood_local_kde(marks)
                output[row_index] = global_likelihood[row_index]
                continue
            group_index = int(group_index)
            variance = variances[group_index]
            if variance.shape[0] != self.n_features:
                raise ValueError("Grouped local mark KDE bandwidth dimensionality does not match the mark features.")
            log_norm = float(np.sum(np.log(2.0 * np.pi * variance)))
            for bin_index in range(self.n_bins):
                support = neighbor_indices[group_index, bin_index]
                weights = log_weights[group_index, bin_index]
                valid = (support >= 0) & np.isfinite(weights)
                if not np.any(valid):
                    diagonal = self._log_mark_likelihood_diagonal_gaussian(
                        marks[row_index : row_index + 1],
                        np.asarray([group_index], dtype=int),
                    )
                    output[row_index, bin_index] = diagonal[0, bin_index]
                    continue
                support_values = kde_values[support[valid]]
                diff = marks[row_index][None, :] - support_values
                quad = np.sum(diff * diff / variance[None, :], axis=1)
                output[row_index, bin_index] = logsumexp(weights[valid] - 0.5 * (quad + log_norm))
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
        *,
        occupancy_s: np.ndarray | None = None,
        return_trajectory: bool = True,
    ) -> EventScore:
        score = super().score(
            emissions,
            bin_centers,
            candidate_indices=candidate_indices,
            occupancy_s=occupancy_s,
            return_trajectory=return_trajectory,
        )
        metadata = getattr(emissions, "metadata", {}) or {}
        score.model_name = str(self.name)
        score.diagnostics["state_space_observation_model"] = "clusterless-marked-point-process"
        score.diagnostics["clusterless_mark_likelihood"] = str(metadata.get("clusterless_mark_likelihood", self.mark_likelihood))
        if "clusterless_mark_kde_bandwidth" in metadata:
            score.diagnostics["clusterless_mark_kde_bandwidth"] = metadata["clusterless_mark_kde_bandwidth"]
        if "clusterless_mark_kde_max_neighbors" in metadata:
            score.diagnostics["clusterless_mark_kde_max_neighbors"] = metadata["clusterless_mark_kde_max_neighbors"]
        if "clusterless_mark_group_by" in metadata:
            score.diagnostics["clusterless_mark_group_by"] = metadata["clusterless_mark_group_by"]
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
    mark_group_by = _normalize_mark_group_by(config.mark_group_by)
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
    if mark_group_ids is not None:
        mark_group_ids = mark_group_ids[keep]
    mark_bins = mark_bins[keep].astype(int)
    finite_rows = np.all(np.isfinite(mark_values), axis=1)
    mark_values = mark_values[finite_rows]
    mark_bins = mark_bins[finite_rows]
    if mark_group_ids is not None:
        mark_group_ids = np.asarray(mark_group_ids[finite_rows], dtype=int)
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
    unique_group_ids = None
    group_rate_hz = None
    group_effective_spike_count = None
    group_mean = None
    group_variance = None
    group_kde_neighbor_indices = None
    group_kde_log_weights = None
    group_kde_variance = None
    if mark_likelihood == _LOCAL_KDE:
        kde_neighbor_indices, kde_log_weights = _local_kde_support(
            mark_bins,
            grid_shape,
            config,
        )
        kde_values = np.asarray(mark_values, dtype=float).copy()
        kde_variance = _mark_kde_variance(mark_values, config)

    if mark_group_by != "none" and mark_group_ids is not None:
        unique_group_ids = np.asarray(sorted(np.unique(mark_group_ids)), dtype=int)
        group_rate_hz, group_effective_spike_count, group_mean, group_variance = _grouped_mark_statistics(
            mark_bins,
            mark_values,
            mark_group_ids,
            unique_group_ids,
            grid_shape,
            smooth_occupancy,
            config,
            encoding_config,
        )
        if mark_likelihood == _LOCAL_KDE:
            group_kde_neighbor_indices, group_kde_log_weights = _grouped_local_kde_support(
                mark_bins,
                mark_group_ids,
                unique_group_ids,
                grid_shape,
                config,
            )
            group_kde_variance = _grouped_mark_kde_variance(mark_values, mark_group_ids, unique_group_ids, config)

    occupancy_denominator = np.maximum(smooth_occupancy, encoding_config.min_occupancy_s)
    rate_hz = np.maximum(smooth_count / occupancy_denominator, config.rate_floor_hz)
    if group_rate_hz is not None:
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
        group_ids=unique_group_ids,
        group_rate_hz=group_rate_hz,
        group_effective_spike_count=group_effective_spike_count,
        group_mark_mean=group_mean,
        group_mark_variance=group_variance,
        group_mark_kde_neighbor_indices=group_kde_neighbor_indices,
        group_mark_kde_log_weights=group_kde_log_weights,
        group_mark_kde_variance=group_kde_variance,
    )


def build_clusterless_mark_emissions(
    session: ReplaySession,
    encoding: ClusterlessMarkEncoding,
    ripple: RippleEvent | int,
    config: EmissionConfig | None = None,
) -> LogEmissionTensor:
    """Build marked-point-process log emissions for one ripple."""

    config = EmissionConfig() if config is None else config
    if not np.isfinite(config.spike_rate_scale) or config.spike_rate_scale <= 0.0:
        raise ValueError("spike_rate_scale must be finite and positive")
    _validate_emission_calibration(
        likelihood_temperature=config.likelihood_temperature,
        negative_binomial_overdispersion=config.negative_binomial_overdispersion,
    )
    if config.cell_weights is not None:
        raise ValueError(
            "cell_weights are only supported for sorted-spike emissions; "
            "use likelihood_temperature to calibrate clusterless emissions"
        )
    if config.negative_binomial_overdispersion > 0.0:
        raise ValueError("negative_binomial_overdispersion is only implemented for sorted-spike emissions")
    marks = session.spike_marks
    if marks is None or marks.n_features == 0:
        raise ValueError("Session does not contain spike marks for clusterless emission scoring.")
    ripple_event = _coerce_ripple_event(session, ripple)
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

    mark_times, mark_values, mark_group_ids = _marks_for_config(session, encoding.config)
    keep = (
        (mark_times >= ripple_event.start)
        & (mark_times < ripple_event.end)
        & np.all(np.isfinite(mark_values), axis=1)
    )
    if np.any(keep):
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        if mark_group_ids is not None:
            mark_group_ids = mark_group_ids[keep]
        time_bins = np.searchsorted(edges, mark_times, side="right") - 1
        valid = (time_bins >= 0) & (time_bins < counts.shape[0])
        time_bins = time_bins[valid].astype(int)
        mark_values = mark_values[valid]
        if mark_group_ids is not None:
            mark_group_ids = np.asarray(mark_group_ids[valid], dtype=int)
        log_rate = np.log(scaled_rate_hz)
        mark_log_likelihood = encoding.log_mark_likelihood(mark_values, mark_group_ids)
        group_indices = encoding._coerce_group_indices(mark_group_ids, mark_values.shape[0]) if mark_group_ids is not None else None
        for local_index, time_bin in enumerate(time_bins):
            local_log_rate = log_rate
            if group_indices is not None and encoding.group_rate_hz is not None and group_indices[local_index] >= 0:
                local_log_rate = np.log(
                    np.maximum(encoding.group_rate_hz[int(group_indices[local_index])] * float(config.spike_rate_scale), np.finfo(float).tiny)
                )
            log_likelihood[time_bin] += local_log_rate + mark_log_likelihood[local_index]
        np.add.at(counts, time_bins, 1)
    log_likelihood += (counts * np.log(bin_durations) - gammaln(counts + 1))[:, None]
    log_likelihood = _apply_likelihood_temperature(log_likelihood, config.likelihood_temperature)

    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=counts[:, None],
        times=times,
        dt=dt,
        cell_ids=np.array([0], dtype=int),
        n_spikes=int(counts.sum()),
        bin_durations=bin_durations,
        transition_durations=np.diff(times) if times.shape[0] > 1 else np.empty(0, dtype=float),
        metadata={
            "clusterless_mark_likelihood": encoding.mark_likelihood,
            "clusterless_mark_kde_bandwidth": _format_float_array(_sqrt_optional(encoding.mark_kde_variance)),
            "clusterless_mark_kde_max_neighbors": _kde_neighbor_count(encoding),
            "clusterless_mark_group_by": _normalize_mark_group_by(encoding.config.mark_group_by),
            "clusterless_mark_groups": encoding.n_mark_groups,
        },
    )
    return emissions


def _validate_mark_config(config: ClusterlessMarkConfig) -> None:
    _normalize_mark_likelihood(config.mark_likelihood)
    _normalize_mark_group_by(config.mark_group_by)

    mark_smoothing_sigma_bins = float(config.mark_smoothing_sigma_bins)
    if not np.isfinite(mark_smoothing_sigma_bins) or mark_smoothing_sigma_bins < 0.0:
        raise ValueError("mark_smoothing_sigma_bins must be nonnegative")
    mark_prior_count = float(config.mark_prior_count)
    if not np.isfinite(mark_prior_count) or mark_prior_count < 0.0:
        raise ValueError("mark_prior_count must be nonnegative")
    mark_variance_floor = float(config.mark_variance_floor)
    if not np.isfinite(mark_variance_floor) or mark_variance_floor <= 0.0:
        raise ValueError("mark_variance_floor must be positive")
    rate_floor_hz = float(config.rate_floor_hz)
    if not np.isfinite(rate_floor_hz) or rate_floor_hz <= 0.0:
        raise ValueError("rate_floor_hz must be positive")

    if config.mark_kde_bandwidth is not None:
        mark_kde_bandwidth = float(config.mark_kde_bandwidth)
        if not np.isfinite(mark_kde_bandwidth) or mark_kde_bandwidth <= 0.0:
            raise ValueError("mark_kde_bandwidth must be positive when provided")
    if config.mark_kde_spatial_sigma_bins is not None:
        mark_kde_spatial_sigma_bins = float(config.mark_kde_spatial_sigma_bins)
        if not np.isfinite(mark_kde_spatial_sigma_bins) or mark_kde_spatial_sigma_bins < 0.0:
            raise ValueError("mark_kde_spatial_sigma_bins must be nonnegative when provided")
    mark_kde_max_neighbors = float(config.mark_kde_max_neighbors)
    if not np.isfinite(mark_kde_max_neighbors) or not mark_kde_max_neighbors.is_integer() or mark_kde_max_neighbors < 1.0:
        raise ValueError("mark_kde_max_neighbors must be a positive integer")


def _normalize_mark_likelihood(value: str) -> str:
    key = str(value).strip().lower().replace("_", "-")
    if key not in _MARK_LIKELIHOOD_ALIASES:
        allowed = ", ".join(sorted({_DIAGONAL_GAUSSIAN, _LOCAL_KDE}))
        raise ValueError(f"Unknown clusterless mark likelihood {value!r}; expected one of: {allowed}")
    return _MARK_LIKELIHOOD_ALIASES[key]


def clusterless_mark_likelihood_label(
    session: ReplaySession,
    mark_likelihood: str | None = None,
) -> str:
    """Return the canonical clusterless mark-likelihood label available for a session."""

    marks = session.spike_marks
    if marks is None or marks.n_features == 0:
        return ""
    if mark_likelihood is None:
        mark_likelihood = ClusterlessMarkConfig().mark_likelihood
    return _normalize_mark_likelihood(mark_likelihood)


def _normalize_mark_group_by(value: str | None) -> str:
    key = "none" if value is None else str(value).strip().lower().replace("_", "-")
    if key not in _MARK_GROUP_BY_ALIASES:
        allowed = ", ".join(sorted(set(_MARK_GROUP_BY_ALIASES.values())))
        raise ValueError(f"Unknown clusterless mark grouping {value!r}; expected one of: {allowed}")
    return _MARK_GROUP_BY_ALIASES[key]


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


def _sqrt_optional(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.sqrt(np.asarray(value, dtype=float))


def _grouped_mark_statistics(
    mark_bins: np.ndarray,
    mark_values: np.ndarray,
    mark_group_ids: np.ndarray,
    unique_group_ids: np.ndarray,
    grid_shape: tuple[int, int],
    smooth_occupancy: np.ndarray,
    config: ClusterlessMarkConfig,
    encoding_config: EncodingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_groups = int(unique_group_ids.shape[0])
    n_bins = int(grid_shape[0] * grid_shape[1])
    n_features = int(mark_values.shape[1])
    group_rates = np.empty((n_groups, n_bins), dtype=float)
    group_counts = np.empty((n_groups, n_bins), dtype=float)
    group_means = np.empty((n_groups, n_bins, n_features), dtype=float)
    group_variances = np.empty_like(group_means)
    prior = max(float(config.mark_prior_count), 0.0)
    occupancy_denominator = np.maximum(smooth_occupancy, encoding_config.min_occupancy_s)
    sigma = float(config.mark_smoothing_sigma_bins)
    for group_index, group_id in enumerate(unique_group_ids):
        mask = mark_group_ids == group_id
        group_values = mark_values[mask]
        group_bins = mark_bins[mask]
        raw_count = np.zeros(n_bins, dtype=float)
        mark_sum = np.zeros((n_features, n_bins), dtype=float)
        mark_sq_sum = np.zeros_like(mark_sum)
        np.add.at(raw_count, group_bins, 1.0)
        for feature_index in range(n_features):
            feature = group_values[:, feature_index]
            np.add.at(mark_sum[feature_index], group_bins, feature)
            np.add.at(mark_sq_sum[feature_index], group_bins, feature * feature)
        if sigma > 0.0:
            smooth_count = gaussian_filter(raw_count.reshape(grid_shape), sigma=sigma, mode="constant").reshape(-1)
            smooth_sum = np.vstack([gaussian_filter(row.reshape(grid_shape), sigma=sigma, mode="constant").reshape(-1) for row in mark_sum])
            smooth_sq_sum = np.vstack([gaussian_filter(row.reshape(grid_shape), sigma=sigma, mode="constant").reshape(-1) for row in mark_sq_sum])
        else:
            smooth_count = raw_count
            smooth_sum = mark_sum
            smooth_sq_sum = mark_sq_sum
        global_mean = np.mean(group_values, axis=0)
        global_variance = np.maximum(np.var(group_values, axis=0), config.mark_variance_floor)
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
        group_counts[group_index] = smooth_count
        group_means[group_index] = mean
        group_variances[group_index] = variance
        group_rates[group_index] = np.maximum(smooth_count / occupancy_denominator, config.rate_floor_hz)
    return group_rates, group_counts, group_means, group_variances


def _grouped_local_kde_support(
    mark_bins: np.ndarray,
    mark_group_ids: np.ndarray,
    unique_group_ids: np.ndarray,
    grid_shape: tuple[int, int],
    config: ClusterlessMarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = int(grid_shape[0] * grid_shape[1])
    group_sizes = [int(np.sum(mark_group_ids == group_id)) for group_id in unique_group_ids]
    max_support = min(int(config.mark_kde_max_neighbors), max(group_sizes))
    neighbor_indices = np.full((unique_group_ids.shape[0], n_bins, max_support), -1, dtype=int)
    log_weights = np.full((unique_group_ids.shape[0], n_bins, max_support), -np.inf, dtype=float)
    for group_index, group_id in enumerate(unique_group_ids):
        global_indices = np.flatnonzero(mark_group_ids == group_id)
        if global_indices.size == 0:
            continue
        local_neighbors, local_log_weights = _local_kde_support(mark_bins[global_indices], grid_shape, config)
        local_support = int(local_neighbors.shape[1])
        neighbor_indices[group_index, :, :local_support] = global_indices[local_neighbors]
        log_weights[group_index, :, :local_support] = local_log_weights
    return neighbor_indices, log_weights


def _grouped_mark_kde_variance(mark_values: np.ndarray, mark_group_ids: np.ndarray, unique_group_ids: np.ndarray, config: ClusterlessMarkConfig) -> np.ndarray:
    return np.vstack([_mark_kde_variance(mark_values[mark_group_ids == group_id], config) for group_id in unique_group_ids])


def _format_float_array(value: np.ndarray | None) -> str:
    if value is None:
        return ""
    return ",".join(f"{float(x):.6g}" for x in np.asarray(value, dtype=float).reshape(-1))


def _kde_neighbor_count(encoding: ClusterlessMarkEncoding) -> int:
    if encoding.mark_kde_neighbor_indices is None:
        return 0
    return int(encoding.mark_kde_neighbor_indices.shape[1])


def _training_marks(
    session: ReplaySession,
    encoding_config: EncodingConfig,
    clusterless_config: ClusterlessMarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    mark_times, mark_values = _all_event_marks(session)
    mark_group_ids = _mark_group_ids_for_config(session, clusterless_config)
    if _use_excitatory_marks(clusterless_config, encoding_config) and marks.cell_ids is not None and session.excitatory_neurons.size:
        keep = np.isin(marks.cell_ids.astype(int), session.excitatory_neurons.astype(int))
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        if mark_group_ids is not None:
            mark_group_ids = mark_group_ids[keep]
    return mark_times, mark_values, mark_group_ids


def _marks_for_config(session: ReplaySession, config: ClusterlessMarkConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    mark_times, mark_values = _all_event_marks(session)
    mark_group_ids = _mark_group_ids_for_config(session, config)
    encoding_config = EncodingConfig() if config.encoding is None else config.encoding
    if _use_excitatory_marks(config, encoding_config) and marks.cell_ids is not None and session.excitatory_neurons.size:
        keep = np.isin(marks.cell_ids.astype(int), session.excitatory_neurons.astype(int))
        mark_times = mark_times[keep]
        mark_values = mark_values[keep]
        if mark_group_ids is not None:
            mark_group_ids = mark_group_ids[keep]
    return mark_times, mark_values, mark_group_ids


def _use_excitatory_marks(
    clusterless_config: ClusterlessMarkConfig,
    encoding_config: EncodingConfig,
) -> bool:
    return bool(clusterless_config.use_excitatory and encoding_config.use_excitatory)


def _mark_group_ids_for_config(session: ReplaySession, config: ClusterlessMarkConfig) -> np.ndarray | None:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    group_by = _normalize_mark_group_by(config.mark_group_by)
    if group_by == "none":
        return None
    if group_by == "cell":
        return None if marks.cell_ids is None else np.asarray(marks.cell_ids, dtype=int)
    if group_by == "tetrode":
        if marks.group_ids is None:
            raise ValueError("clusterless mark grouping by tetrode requires spike-mark group IDs from Tetrode_Cell_IDs")
        return np.asarray(marks.group_ids, dtype=int)
    if marks.group_ids is not None:
        return np.asarray(marks.group_ids, dtype=int)
    return None


def _all_event_marks(session: ReplaySession) -> tuple[np.ndarray, np.ndarray]:
    marks = session.spike_marks
    if marks is None:
        raise ValueError("Session does not contain spike marks.")
    return np.asarray(marks.times, dtype=float), np.asarray(marks.marks, dtype=float)
