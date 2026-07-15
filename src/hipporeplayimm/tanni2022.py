"""NWB I/O and analysis primitives for Tanni, de Cothi, and Barry (2022)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.special import logsumexp


DATASET_DOI = "10.5522/04/18128891.v1"
PAPER_DOI = "10.1016/j.cub.2022.06.046"
ARCHIVE_URL = "https://ndownloader.figshare.com/files/32878716"
ARCHIVE_MD5 = "17b9e752b20edc990d8ed083b3d72c04"
POSITION_PATH = "/acquisition/timeseries/recording1/tracking/ProcessedPos"
LFP_GROUP_PATH = "/acquisition/timeseries/recording1/continuous/processor102_100"
LFP_DATASET_NAME = "downsampled_tetrode_data"
LFP_TIMESTAMPS_NAME = "downsampled_timestamps"
SPIKE_GROUP_PATH = "/acquisition/timeseries/recording1/spikes"
ARENA_SIZE_PATH = "/general/data_collection/Settings/General/arena_size"
ANIMAL_PATH = "/general/data_collection/Settings/General/animal"


@dataclass(frozen=True)
class TanniPosition:
    """Calibrated position and speed in the NWB clock."""

    times_s: np.ndarray
    xy_cm: np.ndarray
    speed_cm_s: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class TanniSpikes:
    """Manually sorted spikes represented as ``(time_s, stable_cell_id)`` rows."""

    spikes: np.ndarray
    cell_ids: np.ndarray
    tetrode_ids: np.ndarray
    cluster_ids: np.ndarray


@dataclass(frozen=True)
class TanniSessionMetadata:
    animal: str
    session: str
    arena_size_cm: np.ndarray
    lfp_sample_rate_hz: float
    n_lfp_channels: int
    lfp_start_time_s: float
    lfp_end_time_s: float


@dataclass(frozen=True)
class RippleCandidate:
    start_time_s: float
    end_time_s: float
    peak_time_s: float
    peak_ripple_z: float

    @property
    def duration_s(self) -> float:
        return float(self.end_time_s - self.start_time_s)


def read_tanni_session_metadata(path: str | Path) -> TanniSessionMetadata:
    """Read lightweight session metadata without loading LFP or waveforms."""

    nwb_path = Path(path)
    with h5py.File(nwb_path, "r") as handle:
        animal = _decode_scalar(handle[ANIMAL_PATH][()])
        arena_size = np.asarray(handle[ARENA_SIZE_PATH][...], dtype=float).reshape(-1)[:2]
        group = handle[LFP_GROUP_PATH]
        lfp = group[LFP_DATASET_NAME]
        timestamps = group[LFP_TIMESTAMPS_NAME]
        sample_rate = _lfp_sample_rate(group, timestamps)
        n_lfp_channels = int(lfp.shape[1])
        start = float(timestamps[0])
        end = float(timestamps[-1])
    return TanniSessionMetadata(
        animal=animal,
        session=nwb_path.parent.name,
        arena_size_cm=arena_size,
        lfp_sample_rate_hz=sample_rate,
        n_lfp_channels=n_lfp_channels,
        lfp_start_time_s=start,
        lfp_end_time_s=end,
    )


def read_tanni_position(path: str | Path, *, speed_smoothing_s: float = 0.1) -> TanniPosition:
    """Read the paper's processed position stream in centimetres.

    The original analysis reads columns 1 and 2 of ``ProcessedPos``. The other
    two coordinate columns are auxiliary LED estimates and are intentionally not
    averaged into the published position estimate.
    """

    with h5py.File(path, "r") as handle:
        data = np.asarray(handle[POSITION_PATH][...], dtype=float)
        arena_size = np.asarray(handle[ARENA_SIZE_PATH][...], dtype=float).reshape(-1)[:2]
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("ProcessedPos must contain timestamp, x, and y columns")
    times = data[:, 0]
    xy = data[:, 1:3]
    keep = np.isfinite(times)
    times = times[keep]
    xy = xy[keep]
    if times.size < 2:
        raise ValueError("ProcessedPos must contain at least two finite timestamps")
    increasing = np.concatenate(([True], np.diff(times) > 0.0))
    times = times[increasing]
    xy = xy[increasing]
    valid = (
        np.all(np.isfinite(xy), axis=1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= arena_size[0])
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= arena_size[1])
    )
    filled = _interpolate_invalid_xy(times, xy, valid)
    dt = float(np.median(np.diff(times)))
    sigma_samples = max(float(speed_smoothing_s) / dt, 0.0)
    smoothed = gaussian_filter1d(filled, sigma=sigma_samples, axis=0, mode="nearest") if sigma_samples > 0.0 else filled
    velocity = np.gradient(smoothed, times, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    speed[~valid] = np.nan
    return TanniPosition(times_s=times, xy_cm=xy, speed_cm_s=speed, valid=valid)


def read_tanni_sorted_spikes(
    path: str | Path,
    *,
    clustering_name: str = "manual_1",
    noise_cluster_id: int = 1,
) -> TanniSpikes:
    """Read manually sorted clusters, applying each tetrode's ``idx_keep`` mask."""

    spike_rows: list[np.ndarray] = []
    tetrode_rows: list[np.ndarray] = []
    cluster_rows: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        group = handle[SPIKE_GROUP_PATH]
        for electrode_name in sorted(group, key=_natural_number):
            electrode = group[electrode_name]
            clustering_path = f"clustering/{clustering_name}"
            if clustering_path not in electrode or "timestamps" not in electrode:
                continue
            tetrode = _natural_number(electrode_name)
            timestamps = np.asarray(electrode["timestamps"][...], dtype=float).reshape(-1)
            labels = np.asarray(electrode[clustering_path][...], dtype=int).reshape(-1)
            if "idx_keep" in electrode:
                keep = np.asarray(electrode["idx_keep"][...], dtype=bool).reshape(-1)
                if keep.shape[0] != timestamps.shape[0]:
                    raise ValueError(f"{electrode_name}: idx_keep and timestamp lengths differ")
                if labels.shape[0] == int(keep.sum()):
                    timestamps = timestamps[keep]
                elif labels.shape[0] != timestamps.shape[0]:
                    raise ValueError(f"{electrode_name}: clustering labels do not align with timestamps")
            elif labels.shape[0] != timestamps.shape[0]:
                raise ValueError(f"{electrode_name}: clustering labels do not align with timestamps")
            retained = np.isfinite(timestamps) & (labels > 0) & (labels != int(noise_cluster_id))
            if not np.any(retained):
                continue
            retained_times = timestamps[retained]
            retained_labels = labels[retained]
            stable_ids = tetrode * 1000 + retained_labels
            spike_rows.append(np.column_stack((retained_times, stable_ids)).astype(float))
            tetrode_rows.append(np.full(retained_times.shape, tetrode, dtype=int))
            cluster_rows.append(retained_labels.astype(int))
    if not spike_rows:
        return TanniSpikes(
            spikes=np.empty((0, 2), dtype=float),
            cell_ids=np.empty(0, dtype=int),
            tetrode_ids=np.empty(0, dtype=int),
            cluster_ids=np.empty(0, dtype=int),
        )
    spikes = np.concatenate(spike_rows, axis=0)
    order = np.argsort(spikes[:, 0], kind="stable")
    spikes = spikes[order]
    cell_ids = np.unique(spikes[:, 1].astype(int))
    return TanniSpikes(
        spikes=spikes,
        cell_ids=cell_ids,
        tetrode_ids=np.concatenate(tetrode_rows),
        cluster_ids=np.concatenate(cluster_rows),
    )


def aggregate_ripple_envelope_z(
    path: str | Path,
    *,
    low_hz: float = 150.0,
    high_hz: float = 250.0,
    envelope_smoothing_s: float = 0.004,
    max_channels: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the mean channelwise ripple-envelope robust z-score.

    Filtering and Hilbert envelopes are computed separately per channel before
    averaging. This prevents a 180-degree LFP phase reversal from cancelling a
    ripple during cross-channel aggregation.
    """

    with h5py.File(path, "r") as handle:
        group = handle[LFP_GROUP_PATH]
        dataset = group[LFP_DATASET_NAME]
        timestamps = np.asarray(group[LFP_TIMESTAMPS_NAME][...], dtype=float)
        sample_rate = _lfp_sample_rate(group, group[LFP_TIMESTAMPS_NAME])
        n_channels = int(dataset.shape[1]) if max_channels is None else min(int(dataset.shape[1]), int(max_channels))
        aggregate = np.zeros(dataset.shape[0], dtype=np.float64)
        for channel_index in range(n_channels):
            signal = np.asarray(dataset[:, channel_index], dtype=float)
            aggregate += ripple_envelope_robust_z(
                signal,
                sample_rate_hz=sample_rate,
                low_hz=low_hz,
                high_hz=high_hz,
                envelope_smoothing_s=envelope_smoothing_s,
            )
    aggregate /= float(n_channels)
    return timestamps, aggregate.astype(np.float32), sample_rate


def ripple_envelope_robust_z(
    signal: np.ndarray,
    *,
    sample_rate_hz: float,
    low_hz: float = 150.0,
    high_hz: float = 250.0,
    envelope_smoothing_s: float = 0.004,
) -> np.ndarray:
    """Band-pass one LFP channel and robust-z its smoothed Hilbert envelope."""

    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size < 32:
        raise ValueError("LFP signal is too short for ripple filtering")
    nyquist = float(sample_rate_hz) / 2.0
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError("ripple band must lie between zero and Nyquist")
    sos = butter(4, (float(low_hz) / nyquist, float(high_hz) / nyquist), btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, values)
    envelope = np.abs(hilbert(filtered))
    sigma = max(float(envelope_smoothing_s) * float(sample_rate_hz), 0.0)
    if sigma > 0.0:
        envelope = gaussian_filter1d(envelope, sigma=sigma, mode="nearest")
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    scale = max(1.4826 * mad, np.finfo(float).eps)
    return (envelope - median) / scale


def detect_ripple_candidates(
    timestamps_s: np.ndarray,
    aggregate_z: np.ndarray,
    *,
    threshold_z: float = 3.0,
    peak_threshold_z: float = 10.0,
    min_duration_s: float = 0.015,
    max_duration_s: float = 0.250,
    merge_gap_s: float = 0.030,
) -> list[RippleCandidate]:
    """Detect threshold crossings and merge nearby ripple-envelope cores."""

    times = np.asarray(timestamps_s, dtype=float).reshape(-1)
    values = np.asarray(aggregate_z, dtype=float).reshape(-1)
    if times.shape != values.shape or times.size < 2:
        raise ValueError("timestamps_s and aggregate_z must be aligned nontrivial vectors")
    above = np.flatnonzero(np.isfinite(values) & (values >= float(threshold_z)))
    if above.size == 0:
        return []
    dt = float(np.median(np.diff(times)))
    max_gap_samples = max(int(round(float(merge_gap_s) / dt)), 0)
    split_points = np.flatnonzero(np.diff(above) > max_gap_samples + 1) + 1
    runs = np.split(above, split_points)
    events: list[RippleCandidate] = []
    for run in runs:
        start_index = int(run[0])
        end_index = int(run[-1])
        duration = float(times[end_index] - times[start_index] + dt)
        if duration < float(min_duration_s) or duration > float(max_duration_s):
            continue
        local = values[start_index : end_index + 1]
        peak_offset = int(np.nanargmax(local))
        peak_index = start_index + peak_offset
        peak_z = float(values[peak_index])
        if peak_z < float(peak_threshold_z):
            continue
        events.append(
            RippleCandidate(
                start_time_s=float(times[start_index]),
                end_time_s=float(times[end_index] + dt),
                peak_time_s=float(times[peak_index]),
                peak_ripple_z=peak_z,
            )
        )
    return events


def nearest_wall_distance(xy_cm: np.ndarray, arena_size_cm: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest wall of a rectangular arena."""

    points = np.asarray(xy_cm, dtype=float)
    arena = np.asarray(arena_size_cm, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or arena.shape[0] < 2:
        raise ValueError("xy_cm must be (n, 2) and arena_size_cm must contain width and height")
    return np.min(
        np.column_stack((points[:, 0], arena[0] - points[:, 0], points[:, 1], arena[1] - points[:, 1])),
        axis=1,
    )


def posterior_from_log_likelihood(log_likelihood: np.ndarray) -> np.ndarray:
    """Normalize independent-bin emission likelihoods into spatial posteriors."""

    log_values = np.asarray(log_likelihood, dtype=float)
    if log_values.ndim != 2 or log_values.shape[0] == 0 or log_values.shape[1] == 0:
        raise ValueError("log_likelihood must be a nonempty two-dimensional array")
    normalized = log_values - logsumexp(log_values, axis=1, keepdims=True)
    return np.exp(normalized)


def posterior_path_segments(
    posterior: np.ndarray,
    bin_centers: np.ndarray,
    rates_hz: np.ndarray,
    occupancy_s: np.ndarray,
    times_s: np.ndarray,
    arena_size_cm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute physical and Poisson code-space speeds for adjacent posterior bins."""

    probabilities = np.asarray(posterior, dtype=float)
    centers = np.asarray(bin_centers, dtype=float)
    rates = np.asarray(rates_hz, dtype=float)
    occupancy = np.asarray(occupancy_s, dtype=float).reshape(-1)
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[1] != centers.shape[0]:
        raise ValueError("posterior and bin_centers do not align")
    if rates.ndim != 2 or rates.shape[1] != centers.shape[0] or occupancy.shape[0] != centers.shape[0]:
        raise ValueError("rates_hz and occupancy_s must align with bin_centers")
    if times.shape[0] != probabilities.shape[0] or times.shape[0] < 2:
        raise ValueError("times_s must align with at least two posterior rows")
    means = probabilities @ centers
    map_positions = centers[np.argmax(probabilities, axis=1)]
    expected_sqrt_rates = probabilities @ np.sqrt(np.maximum(rates, 0.0)).T
    dt = np.diff(times)
    physical_steps = np.linalg.norm(np.diff(means, axis=0), axis=1)
    map_steps = np.linalg.norm(np.diff(map_positions, axis=0), axis=1)
    expected_squared_norm = probabilities @ np.sum(centers * centers, axis=1)
    independent_expected_squared_step = (
        expected_squared_norm[:-1]
        + expected_squared_norm[1:]
        - 2.0 * np.sum(means[:-1] * means[1:], axis=1)
    )
    posterior_rms_steps = np.sqrt(np.maximum(independent_expected_squared_step, 0.0))
    code_steps = np.linalg.norm(np.diff(expected_sqrt_rates, axis=0), axis=1) / np.sqrt(max(rates.shape[0], 1))
    midpoints = 0.5 * (means[:-1] + means[1:])
    map_midpoints = 0.5 * (map_positions[:-1] + map_positions[1:])
    bin_wall_distance = nearest_wall_distance(centers, arena_size_cm)
    expected_wall_distance = probabilities @ bin_wall_distance
    segment_expected_wall_distance = 0.5 * (expected_wall_distance[:-1] + expected_wall_distance[1:])
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)), axis=1)
    centered = centers[None, :, :] - means[:, None, :]
    spread = np.sqrt(np.sum(probabilities * np.sum(centered * centered, axis=2), axis=1))
    local_occupancy = probabilities @ occupancy
    return {
        "time_s": 0.5 * (times[:-1] + times[1:]),
        "x_cm": midpoints[:, 0],
        "y_cm": midpoints[:, 1],
        "wall_distance_cm": segment_expected_wall_distance,
        "wall_distance_normalized": segment_expected_wall_distance / (0.5 * float(np.min(arena_size_cm))),
        "posterior_mean_wall_distance_cm": nearest_wall_distance(midpoints, arena_size_cm),
        "posterior_mean_wall_distance_normalized": nearest_wall_distance(midpoints, arena_size_cm) / (0.5 * float(np.min(arena_size_cm))),
        "map_wall_distance_cm": nearest_wall_distance(map_midpoints, arena_size_cm),
        "physical_speed_cm_s": physical_steps / dt,
        "map_speed_cm_s": map_steps / dt,
        "posterior_rms_independent_speed_cm_s": posterior_rms_steps / dt,
        "code_speed_sqrt_hz_per_s": code_steps / dt,
        "posterior_entropy": 0.5 * (entropy[:-1] + entropy[1:]),
        "posterior_spread_cm": 0.5 * (spread[:-1] + spread[1:]),
        "local_occupancy_s": 0.5 * (local_occupancy[:-1] + local_occupancy[1:]),
    }


def local_poisson_code_gradient(
    bin_centers: np.ndarray,
    rates_hz: np.ndarray,
    *,
    neighbor_radius_cm: float,
) -> np.ndarray:
    """Estimate local Poisson/Hellinger code change per physical centimetre."""

    centers = np.asarray(bin_centers, dtype=float)
    sqrt_rates = np.sqrt(np.maximum(np.asarray(rates_hz, dtype=float), 0.0)).T
    if centers.ndim != 2 or centers.shape[1] != 2 or sqrt_rates.shape[0] != centers.shape[0]:
        raise ValueError("bin_centers and rates_hz do not align")
    output = np.full(centers.shape[0], np.nan, dtype=float)
    for index, center in enumerate(centers):
        distances = np.linalg.norm(centers - center, axis=1)
        neighbors = (distances > 0.0) & (distances <= float(neighbor_radius_cm))
        if not np.any(neighbors):
            continue
        code_distance = np.linalg.norm(sqrt_rates[neighbors] - sqrt_rates[index], axis=1) / np.sqrt(max(rates_hz.shape[0], 1))
        output[index] = float(np.median(code_distance / distances[neighbors]))
    return output


def _lfp_sample_rate(group: h5py.Group, timestamps: h5py.Dataset) -> float:
    if "downsampling_info/downsampled_sampling_rate" in group:
        return float(np.asarray(group["downsampling_info/downsampled_sampling_rate"][()]).reshape(()))
    first = float(timestamps[0])
    last = float(timestamps[min(1000, timestamps.shape[0] - 1)])
    return float(min(1000, timestamps.shape[0] - 1) / (last - first))


def _decode_scalar(value: object) -> str:
    arr = np.asarray(value)
    scalar = arr.reshape(-1)[0] if arr.size else ""
    return scalar.decode("utf-8", errors="replace") if isinstance(scalar, bytes) else str(scalar)


def _natural_number(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else 0


def _interpolate_invalid_xy(times: np.ndarray, xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if np.count_nonzero(valid) < 2:
        raise ValueError("position requires at least two valid samples")
    filled = np.empty_like(xy, dtype=float)
    for dimension in range(2):
        filled[:, dimension] = np.interp(times, times[valid], xy[valid, dimension])
    return filled
