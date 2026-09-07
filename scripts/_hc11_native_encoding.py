"""Native hc-11 encoding helpers, extracted unchanged from commit 6a491825."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class TrackSamples:
    times_s: np.ndarray
    position_cm: np.ndarray
    speed_cm_s: np.ndarray
    direction: np.ndarray
    frame_duration_s: np.ndarray
    maze_mask: np.ndarray
    track_length_cm: float
    topology: str
    maze_type: str
    maze_epoch: np.ndarray
    post_epoch: np.ndarray


@dataclass(frozen=True)
class SpikeData:
    unit_ids: tuple[int, ...]
    times_by_unit: dict[int, np.ndarray]


@dataclass(frozen=True)
class EncodingMap:
    name: str
    unit_ids: tuple[int, ...]
    bin_edges_cm: np.ndarray
    bin_centers_cm: np.ndarray
    occupancy_s: np.ndarray
    prior: np.ndarray
    rates_hz: np.ndarray


def mat_struct(path: Path, variable: str):
    return loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=False)[variable]


def as_intervals(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
    if arr.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    arr = arr[:, :2]
    keep = np.isfinite(arr).all(axis=1) & (arr[:, 1] > arr[:, 0])
    return arr[keep]


def times_in_intervals(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    result = np.zeros(values.shape, dtype=bool)
    for start, end in np.asarray(intervals, dtype=float).reshape(-1, 2):
        result |= (values >= start) & (values <= end)
    return result


def load_track_samples(session_dir: Path) -> TrackSamples:
    base = session_dir.name
    position = mat_struct(session_dir / f"{base}.position.behavior.mat", "position")
    raw_times = np.asarray(position.timestamps, dtype=float).ravel()
    raw_position = np.asarray(position.position.lin, dtype=float).ravel()
    valid = np.isfinite(raw_times) & np.isfinite(raw_position)
    if np.count_nonzero(valid) < 10:
        raise ValueError(f"{base}: too few finite position samples")

    scale = 100.0 if float(np.nanmax(raw_position[valid]) - np.nanmin(raw_position[valid])) < 20.0 else 1.0
    raw_position = raw_position * scale
    maze_epoch = as_intervals(position.Epochs.MazeEpoch)
    post_epoch = as_intervals(position.Epochs.POSTEpoch)
    maze_mask = valid & times_in_intervals(raw_times, maze_epoch)
    if np.count_nonzero(maze_mask) < 10:
        raise ValueError(f"{base}: no finite MAZE position samples")

    maze_type = str(getattr(position.behaviorinfo, "MazeType", "unknown"))
    topology = "circular" if "circular" in maze_type.lower() else "linear"
    lower = float(np.nanmin(raw_position[maze_mask]))
    upper = float(np.nanmax(raw_position[maze_mask]))
    track_length = upper - lower
    if not np.isfinite(track_length) or track_length <= 0.0:
        raise ValueError(f"{base}: invalid track length")
    position_cm = raw_position - lower
    if topology == "circular":
        position_cm = np.mod(position_cm, track_length)
    else:
        position_cm = np.clip(position_cm, 0.0, track_length)

    dt = np.diff(raw_times, append=np.nan)
    finite_dt = dt[np.isfinite(dt) & (dt > 0.0)]
    representative_dt = float(np.median(finite_dt))
    dt[-1] = representative_dt
    dt = np.where(np.isfinite(dt) & (dt > 0.0), np.minimum(dt, 0.25), representative_dt)

    step = np.diff(position_cm, prepend=position_cm[0])
    if topology == "circular":
        step = wrapped_signed_delta(step, track_length)
    velocity = step / np.maximum(np.diff(raw_times, prepend=raw_times[0] - representative_dt), np.finfo(float).eps)
    speed = np.abs(velocity)
    direction = np.zeros(velocity.shape, dtype=int)
    finite_velocity = np.isfinite(velocity)
    direction[finite_velocity] = np.sign(velocity[finite_velocity]).astype(int)
    speed[~np.isfinite(speed)] = np.nan
    return TrackSamples(
        times_s=raw_times,
        position_cm=position_cm,
        speed_cm_s=speed,
        direction=direction,
        frame_duration_s=dt,
        maze_mask=maze_mask,
        track_length_cm=track_length,
        topology=topology,
        maze_type=maze_type,
        maze_epoch=maze_epoch,
        post_epoch=post_epoch,
    )


def load_spikes(session_dir: Path) -> SpikeData:
    base = session_dir.name
    spikes = mat_struct(session_dir / f"{base}.spikes.cellinfo.mat", "spikes")
    unit_ids = np.asarray(spikes.UID, dtype=int).ravel()
    times_cells = np.asarray(spikes.times, dtype=object).ravel()
    regions = np.asarray(spikes.region, dtype=object).ravel()
    times_by_unit: dict[int, np.ndarray] = {}
    for index, unit_id in enumerate(unit_ids):
        region = str(regions[index]) if index < regions.size else ""
        if "CA1" not in region.upper():
            continue
        values = np.asarray(times_cells[index], dtype=float).ravel()
        values = np.sort(values[np.isfinite(values)])
        if values.size:
            times_by_unit[int(unit_id)] = values
    return SpikeData(tuple(sorted(times_by_unit)), times_by_unit)


def nearest_frame_indices(frame_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(frame_times, query_times, side="left")
    right = np.clip(right, 0, len(frame_times) - 1)
    left = np.clip(right - 1, 0, len(frame_times) - 1)
    use_left = np.abs(query_times - frame_times[left]) <= np.abs(frame_times[right] - query_times)
    return np.where(use_left, left, right)


def make_bin_edges(track_length_cm: float, bin_size_cm: float) -> np.ndarray:
    n_bins = max(int(np.ceil(float(track_length_cm) / float(bin_size_cm))), 2)
    return np.linspace(0.0, float(track_length_cm), n_bins + 1)


def fit_encoding_map(
    track: TrackSamples,
    spikes: SpikeData,
    unit_ids: tuple[int, ...],
    *,
    frame_mask: np.ndarray,
    bin_edges_cm: np.ndarray,
    smoothing_sigma_bins: float,
    name: str,
) -> EncodingMap:
    frame_mask = np.asarray(frame_mask, dtype=bool) & track.maze_mask & np.isfinite(track.position_cm)
    occupancy, _ = np.histogram(
        track.position_cm[frame_mask],
        bins=bin_edges_cm,
        weights=track.frame_duration_s[frame_mask],
    )
    counts = np.zeros((len(unit_ids), len(bin_edges_cm) - 1), dtype=float)
    for row, unit_id in enumerate(unit_ids):
        unit_times = spikes.times_by_unit[int(unit_id)]
        in_maze = times_in_intervals(unit_times, track.maze_epoch)
        selected_times = unit_times[in_maze]
        if selected_times.size == 0:
            continue
        frame_indices = nearest_frame_indices(track.times_s, selected_times)
        keep = frame_mask[frame_indices]
        counts[row], _ = np.histogram(track.position_cm[frame_indices[keep]], bins=bin_edges_cm)

    mode = "wrap" if track.topology == "circular" else "nearest"
    smooth_occupancy = gaussian_filter1d(occupancy.astype(float), smoothing_sigma_bins, mode=mode)
    smooth_counts = gaussian_filter1d(counts, smoothing_sigma_bins, axis=1, mode=mode)
    rates = smooth_counts / np.maximum(smooth_occupancy[None, :], 1e-6)
    rates = np.maximum(rates, 1e-4)
    prior = smooth_occupancy / max(float(smooth_occupancy.sum()), np.finfo(float).eps)
    centers = 0.5 * (bin_edges_cm[:-1] + bin_edges_cm[1:])
    return EncodingMap(name, unit_ids, bin_edges_cm, centers, smooth_occupancy, prior, rates)


def spatial_information_bits_per_spike(encoding: EncodingMap) -> np.ndarray:
    occupancy_probability = encoding.occupancy_s / max(float(encoding.occupancy_s.sum()), np.finfo(float).eps)
    mean_rate = encoding.rates_hz @ occupancy_probability
    ratio = encoding.rates_hz / np.maximum(mean_rate[:, None], np.finfo(float).eps)
    terms = occupancy_probability[None, :] * ratio * np.log2(np.maximum(ratio, np.finfo(float).eps))
    return np.sum(terms, axis=1)


def build_session_encodings(
    track: TrackSamples,
    spikes: SpikeData,
    *,
    position_bin_size_cm: float,
    min_run_speed_cm_s: float,
    min_run_spikes: int,
    min_spatial_information: float,
    min_peak_rate_hz: float,
    min_encoding_units: int,
    smoothing_sigma_bins: float,
) -> tuple[dict[str, list[EncodingMap]], pd.DataFrame]:
    edges = make_bin_edges(track.track_length_cm, position_bin_size_cm)
    moving = track.maze_mask & (track.speed_cm_s >= float(min_run_speed_cm_s))
    preliminary = fit_encoding_map(
        track,
        spikes,
        spikes.unit_ids,
        frame_mask=moving,
        bin_edges_cm=edges,
        smoothing_sigma_bins=smoothing_sigma_bins,
        name="pooled",
    )
    information = spatial_information_bits_per_spike(preliminary)
    rows: list[dict[str, object]] = []
    selected: list[int] = []
    for index, unit_id in enumerate(spikes.unit_ids):
        run_spikes = int(np.rint(np.sum(preliminary.rates_hz[index] * preliminary.occupancy_s)))
        peak_rate = float(np.max(preliminary.rates_hz[index]))
        passed = run_spikes >= min_run_spikes and information[index] >= min_spatial_information and peak_rate >= min_peak_rate_hz
        rows.append(
            {
                "unit_id": int(unit_id),
                "run_spikes": run_spikes,
                "peak_rate_hz": peak_rate,
                "spatial_information_bits_per_spike": float(information[index]),
                "unit_qc_passed": bool(passed),
            }
        )
        if passed:
            selected.append(int(unit_id))
    if len(selected) < int(min_encoding_units):
        raise ValueError(f"only {len(selected)} place-like units pass QC; need {min_encoding_units}")
    selected_units = tuple(selected)
    pooled = fit_encoding_map(
        track,
        spikes,
        selected_units,
        frame_mask=moving,
        bin_edges_cm=edges,
        smoothing_sigma_bins=smoothing_sigma_bins,
        name="pooled",
    )
    positive = fit_encoding_map(
        track,
        spikes,
        selected_units,
        frame_mask=moving & (track.direction > 0),
        bin_edges_cm=edges,
        smoothing_sigma_bins=smoothing_sigma_bins,
        name="positive_direction",
    )
    negative = fit_encoding_map(
        track,
        spikes,
        selected_units,
        frame_mask=moving & (track.direction < 0),
        bin_edges_cm=edges,
        smoothing_sigma_bins=smoothing_sigma_bins,
        name="negative_direction",
    )
    maps = {"pooled": [pooled], "direction_mixture": [negative, positive]}
    return maps, pd.DataFrame(rows)


def wrapped_signed_delta(delta: np.ndarray | float, track_length_cm: float) -> np.ndarray:
    values = np.asarray(delta, dtype=float)
    return (values + 0.5 * track_length_cm) % track_length_cm - 0.5 * track_length_cm


def topology_distance(left: np.ndarray, right: np.ndarray, topology: str, track_length_cm: float) -> np.ndarray:
    distance = np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
    if topology == "circular":
        distance = np.minimum(distance, track_length_cm - distance)
    return distance


def topology_gaussian_transition(
    bin_centers_cm: np.ndarray,
    sigma_cm: float,
    max_step_sigma: float,
    *,
    topology: str,
    track_length_cm: float,
) -> csr_matrix:
    centers = np.asarray(bin_centers_cm, dtype=float).ravel()
    if topology not in {"linear", "circular"}:
        raise ValueError("topology must be linear or circular")
    if not np.isfinite(sigma_cm) or sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be finite and positive")
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    radius = float(sigma_cm) * float(max_step_sigma)
    for source, center in enumerate(centers):
        distances = topology_distance(centers, center, topology, track_length_cm)
        keep = distances <= radius
        if not np.any(keep):
            keep[int(np.argmin(distances))] = True
        destinations = np.flatnonzero(keep)
        weights = np.exp(-0.5 * np.square(distances[destinations] / sigma_cm))
        weights /= float(weights.sum())
        rows.extend(int(value) for value in destinations)
        cols.extend([source] * len(destinations))
        data.extend(float(value) for value in weights)
    return csr_matrix((data, (rows, cols)), shape=(len(centers), len(centers)))
