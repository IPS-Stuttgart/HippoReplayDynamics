#!/usr/bin/env python3
"""Score a balanced native-ripple hc-11 pilot with topology-aware 1D models.

The public hc-11 Webshare sessions mix linear and circular mazes.  This script
keeps those geometries explicit: linear transitions reflect at track ends,
whereas circular transitions wrap across the coordinate seam.  It also reports
a direction-conditioned sensitivity so pooled clockwise/counter-clockwise or
inbound/outbound fields cannot silently create an apparent switching-model win.

This is a native-ripple technical/generalization pilot.  It does not promote a
biological IMM claim; later Gate 2/3/4 controls remain required for that.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zlib

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
import hipporeplayimm.state_space as state_space  # noqa: E402
from hipporeplayimm.duration_occupancy import (  # noqa: E402
    _forward_backward_variable,
    _score_first_order_imm_variable,
)
from hipporeplayimm.encoding import LogEmissionTensor  # noqa: E402
from hipporeplayimm.state_space_first_order import (  # noqa: E402
    _score_fragmented,
    _score_stationary,
)


DEFAULT_DATASET_ROOT = Path("/mnt/lexar4tb/datasets/hc11_grosmark_buzsaki/webshare_processed")
DEFAULT_OUTPUT_DIR = Path("results/hc11-native-ripple-geometry-pilot")

EVIDENCE_OUTPUT = "hc11_native_ripple_event_model_evidence.csv"
DECISION_OUTPUT = "hc11_native_ripple_model_claim_decisions.csv"
DECODER_OUTPUT = "hc11_native_ripple_decoder_qc.csv"
UNIT_OUTPUT = "hc11_native_ripple_encoding_unit_qc.csv"
SELECTION_OUTPUT = "hc11_native_ripple_event_selection.csv"
SESSION_OUTPUT = "hc11_native_ripple_by_session.csv"
ANIMAL_OUTPUT = "hc11_native_ripple_by_animal.csv"
GEOMETRY_OUTPUT = "hc11_native_ripple_by_geometry.csv"
DIRECTION_OUTPUT = "hc11_native_ripple_direction_sensitivity.csv"
GATE_OUTPUT = "hc11_native_ripple_gate_summary.csv"
MANIFEST_OUTPUT = "hc11_native_ripple_manifest.json"
SUMMARY_OUTPUT = "hc11_native_ripple_summary.md"

MODELS = ("stationary", "diffusion", "fragmented", "first_order_imm")
TRAJECTORY_MODELS = ("diffusion", "fragmented", "first_order_imm")
PRIMARY_ENCODING_VARIANT = "direction_mixture"


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


def discover_native_ripple_sessions(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.glob("*/*") if path.is_dir() and list(path.glob("*.ripplesNREM.event.mat")))


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


def poisson_log_likelihood(counts: np.ndarray, rates_hz: np.ndarray, durations_s: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    rates = np.maximum(np.asarray(rates_hz, dtype=float), 1e-12)
    durations = np.asarray(durations_s, dtype=float)
    log_rates = np.log(rates)
    spike_term = counts @ log_rates
    duration_term = durations[:, None] * np.sum(rates, axis=0)[None, :]
    constants = np.sum(gammaln(counts + 1.0), axis=1) - np.sum(counts, axis=1) * np.log(durations)
    return spike_term - duration_term - constants[:, None]


def event_bin_edges(start_s: float, end_s: float, time_bin_s: float) -> np.ndarray:
    edges = np.arange(float(start_s), float(end_s), float(time_bin_s))
    if edges.size == 0 or not np.isclose(edges[0], start_s):
        edges = np.insert(edges, 0, float(start_s))
    if edges[-1] < end_s:
        edges = np.append(edges, float(end_s))
    if edges.size < 2:
        edges = np.array([float(start_s), float(end_s)])
    return edges


def spike_count_matrix(spikes: SpikeData, unit_ids: tuple[int, ...], edges: np.ndarray) -> np.ndarray:
    counts = np.zeros((len(edges) - 1, len(unit_ids)), dtype=int)
    for column, unit_id in enumerate(unit_ids):
        counts[:, column], _ = np.histogram(spikes.times_by_unit[int(unit_id)], bins=edges)
    return counts


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


def score_single_encoding(
    counts: np.ndarray,
    edges: np.ndarray,
    encoding: EncodingMap,
    *,
    topology: str,
    track_length_cm: float,
    diffusion_sigma_cm_sqrt_s: float,
    stationary_sigma_cm: float,
    max_step_sigma: float,
    imm_mode_stickiness: float,
) -> dict[str, dict[str, object]]:
    durations = np.diff(edges)
    centers_time = 0.5 * (edges[:-1] + edges[1:])
    transition_durations = np.diff(centers_time)
    log_likelihood = poisson_log_likelihood(counts, encoding.rates_hz, durations)
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=counts,
        times=centers_time,
        dt=float(np.median(durations)),
        cell_ids=np.asarray(encoding.unit_ids, dtype=int),
        n_spikes=int(counts.sum()),
        bin_durations=durations,
        transition_durations=transition_durations,
    )
    centers = encoding.bin_centers_cm.reshape(-1, 1)
    stationary_transition = topology_gaussian_transition(
        encoding.bin_centers_cm,
        stationary_sigma_cm,
        max_step_sigma,
        topology=topology,
        track_length_cm=track_length_cm,
    )
    diffusion_transitions = [
        topology_gaussian_transition(
            encoding.bin_centers_cm,
            diffusion_sigma_cm_sqrt_s * np.sqrt(float(duration)),
            max_step_sigma,
            topology=topology,
            track_length_cm=track_length_cm,
        )
        for duration in transition_durations
    ]
    stationary_logz, stationary_posterior = _score_stationary(emissions)
    fragmented_logz, fragmented_posterior = _score_fragmented(emissions)
    if transition_durations.size:
        diffusion_logz, diffusion_posterior = _forward_backward_variable(
            state_space,
            log_likelihood,
            diffusion_transitions,
        )
        imm_logz, imm_posterior, mode_posterior = _score_first_order_imm_variable(
            state_space,
            log_likelihood,
            centers,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_transitions=diffusion_transitions,
            max_step_sigma=max_step_sigma,
            mode_stickiness=imm_mode_stickiness,
            stationary_transitions=stationary_transition,
        )
    else:
        diffusion_logz, diffusion_posterior = fragmented_logz, fragmented_posterior
        imm_logz, imm_posterior = fragmented_logz, fragmented_posterior
        mode_posterior = np.full((len(counts), 3), 1.0 / 3.0)
    return {
        "stationary": {"log_evidence": stationary_logz, "posterior": stationary_posterior},
        "diffusion": {"log_evidence": diffusion_logz, "posterior": diffusion_posterior},
        "fragmented": {"log_evidence": fragmented_logz, "posterior": fragmented_posterior},
        "first_order_imm": {"log_evidence": imm_logz, "posterior": imm_posterior, "mode_posterior": mode_posterior},
    }


def posterior_expected_position(posterior: np.ndarray, centers: np.ndarray, topology: str, track_length_cm: float) -> np.ndarray:
    probabilities = np.asarray(posterior, dtype=float)
    if topology == "linear":
        return probabilities @ centers
    angles = 2.0 * np.pi * centers / track_length_cm
    sine = probabilities @ np.sin(angles)
    cosine = probabilities @ np.cos(angles)
    return np.mod(np.arctan2(sine, cosine), 2.0 * np.pi) * track_length_cm / (2.0 * np.pi)


def imm_content_diagnostics(
    posterior: np.ndarray,
    mode_posterior: np.ndarray,
    centers: np.ndarray,
    topology: str,
    track_length_cm: float,
    duration_s: float,
) -> dict[str, float]:
    expected = posterior_expected_position(posterior, centers, topology, track_length_cm)
    steps = np.diff(expected)
    if topology == "circular":
        steps = wrapped_signed_delta(steps, track_length_cm)
        net = abs(float(wrapped_signed_delta(expected[-1] - expected[0], track_length_cm))) if len(expected) else 0.0
    else:
        net = abs(float(expected[-1] - expected[0])) if len(expected) else 0.0
    map_mode = np.argmax(mode_posterior, axis=1)
    return {
        "mean_stationary_mode_probability": float(np.mean(mode_posterior[:, 0])),
        "mean_nonstationary_mode_probability": float(np.mean(mode_posterior[:, 1:].sum(axis=1))),
        "fraction_time_map_nonstationary": float(np.mean(map_mode != 0)),
        "posterior_expected_path_length_cm": float(np.sum(np.abs(steps))),
        "posterior_net_displacement_cm": net,
        "posterior_path_speed_cm_s": float(np.sum(np.abs(steps)) / max(duration_s, np.finfo(float).eps)),
    }


def score_encoding_variant(
    counts: np.ndarray,
    edges: np.ndarray,
    encodings: list[EncodingMap],
    **model_kwargs,
) -> dict[str, dict[str, object]]:
    map_scores = [score_single_encoding(counts, edges, encoding, **model_kwargs) for encoding in encodings]
    combined: dict[str, dict[str, object]] = {}
    for model in MODELS:
        logz_values = np.asarray([float(scores[model]["log_evidence"]) for scores in map_scores])
        combined_logz = float(logsumexp(logz_values) - np.log(len(logz_values)))
        map_weights = np.exp(logz_values - logsumexp(logz_values))
        posterior = sum(weight * np.exp(scores[model]["posterior"]) for weight, scores in zip(map_weights, map_scores, strict=True))
        result: dict[str, object] = {"log_evidence": combined_logz, "posterior": posterior}
        if model == "first_order_imm":
            result["mode_posterior"] = sum(
                weight * np.asarray(scores[model]["mode_posterior"], dtype=float)
                for weight, scores in zip(map_weights, map_scores, strict=True)
            )
        combined[model] = result
    return combined


def load_native_post_nrem_events(
    session_dir: Path,
    track: TrackSamples,
    spikes: SpikeData,
    unit_ids: tuple[int, ...],
    *,
    min_event_spikes: int,
    min_event_active_units: int,
    max_events: int,
    event_ranking: str,
    selection_seed: int,
) -> pd.DataFrame:
    base = session_dir.name
    sleep_state = mat_struct(session_dir / f"{base}.SleepState.states.mat", "SleepState")
    nrem = as_intervals(sleep_state.ints.NREMstate)
    with h5py.File(session_dir / f"{base}.ripplesNREM.event.mat", "r") as handle:
        group = handle["ripplesNREM"]
        times = np.asarray(group["times"], dtype=float)
        peaks = np.asarray(group["peaks"], dtype=float).ravel() if "peaks" in group else np.array([])
        powers = np.asarray(group["peakNormedPower"], dtype=float).ravel() if "peakNormedPower" in group else np.array([])
    if times.shape[0] != 2 and times.shape[-1] == 2:
        times = times.T
    rows: list[dict[str, object]] = []
    for event_id, (start_s, end_s) in enumerate(zip(times[0], times[1], strict=True)):
        peak_s = float(peaks[event_id]) if event_id < len(peaks) and np.isfinite(peaks[event_id]) else 0.5 * (start_s + end_s)
        if not times_in_intervals(np.array([peak_s]), track.post_epoch)[0] or not times_in_intervals(np.array([peak_s]), nrem)[0]:
            continue
        counts = []
        for unit_id in unit_ids:
            values = spikes.times_by_unit[int(unit_id)]
            left = np.searchsorted(values, start_s, side="left")
            right = np.searchsorted(values, end_s, side="right")
            counts.append(int(right - left))
        n_spikes = int(np.sum(counts))
        n_active = int(np.sum(np.asarray(counts) > 0))
        if n_spikes < min_event_spikes or n_active < min_event_active_units:
            continue
        rows.append(
            {
                "event_id": int(event_id),
                "start_time_s": float(start_s),
                "end_time_s": float(end_s),
                "duration_ms": float(1000.0 * (end_s - start_s)),
                "peak_time_s": peak_s,
                "peak_ripple_power_z": float(powers[event_id]) if event_id < len(powers) and np.isfinite(powers[event_id]) else np.nan,
                "n_spikes": n_spikes,
                "n_active_units": n_active,
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return rank_native_events(
        events,
        event_ranking=event_ranking,
        max_events=max_events,
        selection_seed=selection_seed,
        session_name=session_dir.name,
    )


def rank_native_events(
    events: pd.DataFrame,
    *,
    event_ranking: str,
    max_events: int,
    selection_seed: int,
    session_name: str,
) -> pd.DataFrame:
    if event_ranking == "peak_ripple_power":
        ranked = events.sort_values(
            ["peak_ripple_power_z", "n_active_units", "n_spikes", "event_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
            na_position="last",
        )
        score_name = "peak_ripple_power_z"
        score_values = ranked["peak_ripple_power_z"]
    elif event_ranking == "spike_support":
        ranked = events.sort_values(
            ["n_active_units", "n_spikes", "peak_ripple_power_z", "event_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
            na_position="last",
        )
        score_name = "n_active_units_then_n_spikes"
        score_values = ranked["n_active_units"].astype(float) * 1_000_000.0 + ranked["n_spikes"].astype(float)
    elif event_ranking == "random":
        stable_session_seed = zlib.crc32(session_name.encode("utf-8"))
        rng = np.random.default_rng(int(selection_seed) + int(stable_session_seed))
        ranked = events.assign(_random_rank=rng.random(len(events))).sort_values(
            ["_random_rank", "event_id"],
            ascending=[True, True],
            kind="mergesort",
        )
        score_name = "deterministic_random_rank"
        score_values = -ranked["_random_rank"]
    else:
        raise ValueError("event_ranking must be peak_ripple_power, spike_support, or random")
    if max_events > 0:
        ranked = ranked.head(max_events).copy()
        score_values = score_values.loc[ranked.index]
    ranked["selection_rule"] = event_ranking
    ranked["selection_score_name"] = score_name
    ranked["selection_score_value"] = np.asarray(score_values, dtype=float)
    ranked.insert(0, "selection_rank_within_session", np.arange(1, len(ranked) + 1))
    return ranked.drop(columns=["_random_rank"], errors="ignore").reset_index(drop=True)


def decode_crossvalidated(
    track: TrackSamples,
    spikes: SpikeData,
    unit_ids: tuple[int, ...],
    *,
    position_bin_size_cm: float,
    min_run_speed_cm_s: float,
    smoothing_sigma_bins: float,
    n_folds: int,
    decode_window_s: float,
    max_decode_bins: int,
) -> dict[str, float | int | bool]:
    moving_indices = np.flatnonzero(track.maze_mask & (track.speed_cm_s >= min_run_speed_cm_s) & (track.direction != 0))
    if moving_indices.size < n_folds:
        return {"decoder_status": "insufficient_moving_position", "decoder_qc_passed": False}
    fold_edges = np.linspace(0, moving_indices.size, n_folds + 1, dtype=int)
    edges = make_bin_edges(track.track_length_cm, position_bin_size_cm)
    errors: dict[str, list[float]] = {"pooled_map": [], "direction_observed": []}
    sample_count = 0
    for fold in range(n_folds):
        test_indices = moving_indices[fold_edges[fold] : fold_edges[fold + 1]]
        train_mask = track.maze_mask & (track.speed_cm_s >= min_run_speed_cm_s)
        train_mask[test_indices] = False
        maps = {
            "pooled_map": fit_encoding_map(track, spikes, unit_ids, frame_mask=train_mask, bin_edges_cm=edges, smoothing_sigma_bins=smoothing_sigma_bins, name="pooled"),
            "negative_direction": fit_encoding_map(
                track,
                spikes,
                unit_ids,
                frame_mask=train_mask & (track.direction < 0),
                bin_edges_cm=edges,
                smoothing_sigma_bins=smoothing_sigma_bins,
                name="negative_direction",
            ),
            "positive_direction": fit_encoding_map(
                track,
                spikes,
                unit_ids,
                frame_mask=train_mask & (track.direction > 0),
                bin_edges_cm=edges,
                smoothing_sigma_bins=smoothing_sigma_bins,
                name="positive_direction",
            ),
        }
        stride = max(int(round(decode_window_s / np.nanmedian(track.frame_duration_s[test_indices]))), 1)
        test_indices = test_indices[::stride]
        if max_decode_bins > 0 and len(test_indices) > max_decode_bins // n_folds:
            chosen = np.linspace(0, len(test_indices) - 1, max_decode_bins // n_folds, dtype=int)
            test_indices = test_indices[chosen]
        for frame_index in test_indices:
            center_time = track.times_s[frame_index]
            event_edges = np.array([center_time - 0.5 * decode_window_s, center_time + 0.5 * decode_window_s])
            counts = spike_count_matrix(spikes, unit_ids, event_edges)
            for variant in errors:
                encoding = maps["pooled_map"] if variant == "pooled_map" else maps[
                    "positive_direction" if track.direction[frame_index] > 0 else "negative_direction"
                ]
                log_likelihood = poisson_log_likelihood(counts, encoding.rates_hz, np.array([decode_window_s]))[0]
                log_posterior = log_likelihood + np.log(np.maximum(encoding.prior, 1e-300))
                posterior = np.exp(log_posterior - logsumexp(log_posterior))
                decoded = posterior_expected_position(
                    posterior[None, :],
                    encoding.bin_centers_cm,
                    track.topology,
                    track.track_length_cm,
                )[0]
                error = topology_distance(decoded, track.position_cm[frame_index], track.topology, track.track_length_cm)
                errors[variant].append(float(error))
            sample_count += 1
    pooled = np.asarray(errors["pooled_map"], dtype=float)
    directional = np.asarray(errors["direction_observed"], dtype=float)
    pooled_median = float(np.nanmedian(pooled)) if pooled.size else np.nan
    directional_median = float(np.nanmedian(directional)) if directional.size else np.nan
    passed = bool(np.isfinite(pooled_median) and pooled_median <= min(35.0, 0.25 * track.track_length_cm) and sample_count > 0)
    return {
        "decoder_status": "pass" if passed else "finite_but_above_threshold",
        "decoder_qc_passed": passed,
        "crossval_n_folds": int(n_folds),
        "crossval_samples": int(sample_count),
        "pooled_map_error_cm_median": pooled_median,
        "pooled_map_error_cm_p75": float(np.nanpercentile(pooled, 75)) if pooled.size else np.nan,
        "direction_observed_error_cm_median": directional_median,
        "direction_observed_error_cm_p75": float(np.nanpercentile(directional, 75)) if directional.size else np.nan,
        "direction_conditioning_error_delta_cm": pooled_median - directional_median,
        "decoder_error_threshold_cm": min(35.0, 0.25 * track.track_length_cm),
    }


def event_decisions(evidence: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    primary = evidence[(evidence["encoding_variant"] == PRIMARY_ENCODING_VARIANT) & (evidence["status"] == "success")]
    rows: list[dict[str, object]] = []
    keys = ["animal", "session", "geometry", "maze_type", "event_id"]
    for key, group in primary.groupby(keys, sort=True):
        logz = dict(zip(group["model"], group["log_evidence"], strict=True))
        if not all(model in logz for model in MODELS):
            continue
        ordered = sorted(logz.items(), key=lambda item: item[1], reverse=True)
        best_trajectory = max(logz[model] for model in TRAJECTORY_MODELS)
        metadata = group.iloc[0]
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "selection_rank_within_session": int(metadata["selection_rank_within_session"]),
                "duration_ms": float(metadata["duration_ms"]),
                "raw_ripple_duration_ms": float(metadata["raw_ripple_duration_ms"]),
                "n_spikes": int(metadata["n_spikes"]),
                "raw_ripple_n_spikes": int(metadata["raw_ripple_n_spikes"]),
                "n_active_units": int(metadata["n_active_units"]),
                "n_time_bins": int(metadata["n_time_bins"]),
                "best_model": ordered[0][0],
                "runner_up_model": ordered[1][0],
                "best_minus_runner_up_log_evidence": float(ordered[0][1] - ordered[1][1]),
                **{f"logZ_{model}": float(logz[model]) for model in MODELS},
                "delta_trajectory_minus_stationary": float(best_trajectory - logz["stationary"]),
                "delta_imm_minus_fragmented": float(logz["first_order_imm"] - logz["fragmented"]),
                "trajectory_confident_claim": bool(best_trajectory - logz["stationary"] >= margin_threshold),
                "stationary_confident_claim": bool(logz["stationary"] - best_trajectory >= margin_threshold),
                "imm_confident_over_fragmented": bool(logz["first_order_imm"] - logz["fragmented"] >= margin_threshold),
                "fragmented_confident_over_imm": bool(logz["fragmented"] - logz["first_order_imm"] >= margin_threshold),
            }
        )
        row["delta_trajectory_minus_stationary_per_time_bin"] = row["delta_trajectory_minus_stationary"] / max(row["n_time_bins"], 1)
        row["delta_trajectory_minus_stationary_per_spike"] = row["delta_trajectory_minus_stationary"] / max(row["n_spikes"], 1)
        row["delta_imm_minus_fragmented_per_time_bin"] = row["delta_imm_minus_fragmented"] / max(row["n_time_bins"], 1)
        row["delta_imm_minus_fragmented_per_spike"] = row["delta_imm_minus_fragmented"] / max(row["n_spikes"], 1)
        imm_row = group[group["model"] == "first_order_imm"].iloc[0]
        for column in (
            "mean_stationary_mode_probability",
            "mean_nonstationary_mode_probability",
            "fraction_time_map_nonstationary",
            "posterior_expected_path_length_cm",
            "posterior_net_displacement_cm",
            "posterior_path_speed_cm_s",
        ):
            row[column] = float(imm_row[column])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_decisions(decisions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    iterator = decisions.groupby(group_columns, sort=True) if group_columns else [((), decisions)]
    for keys, group in iterator:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        best = group["best_model"].value_counts()
        row.update(
            {
                "events": int(len(group)),
                "trajectory_confident_count": int(group["trajectory_confident_claim"].sum()),
                "trajectory_confident_fraction": float(group["trajectory_confident_claim"].mean()),
                "stationary_confident_count": int(group["stationary_confident_claim"].sum()),
                "median_trajectory_minus_stationary": float(group["delta_trajectory_minus_stationary"].median()),
                "median_trajectory_minus_stationary_per_time_bin": float(group["delta_trajectory_minus_stationary_per_time_bin"].median()),
                "median_trajectory_minus_stationary_per_spike": float(group["delta_trajectory_minus_stationary_per_spike"].median()),
                "median_imm_minus_fragmented": float(group["delta_imm_minus_fragmented"].median()),
                "median_imm_minus_fragmented_per_time_bin": float(group["delta_imm_minus_fragmented_per_time_bin"].median()),
                "median_imm_minus_fragmented_per_spike": float(group["delta_imm_minus_fragmented_per_spike"].median()),
                "imm_confident_over_fragmented_count": int(group["imm_confident_over_fragmented"].sum()),
                "fragmented_confident_over_imm_count": int(group["fragmented_confident_over_imm"].sum()),
                "stationary_best_count": int(best.get("stationary", 0)),
                "diffusion_best_count": int(best.get("diffusion", 0)),
                "fragmented_best_count": int(best.get("fragmented", 0)),
                "first_order_imm_best_count": int(best.get("first_order_imm", 0)),
                "median_nonstationary_mode_probability": float(group["mean_nonstationary_mode_probability"].median()),
                "median_posterior_path_length_cm": float(group["posterior_expected_path_length_cm"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def direction_sensitivity(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["animal", "session", "geometry", "event_id"]
    for key, group in evidence[evidence["status"] == "success"].groupby(keys, sort=True):
        values = group.pivot_table(index="encoding_variant", columns="model", values="log_evidence", aggfunc="first")
        if not {"pooled", "direction_mixture"}.issubset(values.index):
            continue
        row = dict(zip(keys, key, strict=True))
        for variant in ("pooled", "direction_mixture"):
            trajectory = max(float(values.loc[variant, model]) for model in TRAJECTORY_MODELS)
            row[f"{variant}_trajectory_minus_stationary"] = trajectory - float(values.loc[variant, "stationary"])
            row[f"{variant}_imm_minus_fragmented"] = float(values.loc[variant, "first_order_imm"] - values.loc[variant, "fragmented"])
        row["direction_conditioning_delta_trajectory_margin"] = row["direction_mixture_trajectory_minus_stationary"] - row["pooled_trajectory_minus_stationary"]
        row["direction_conditioning_delta_imm_margin"] = row["direction_mixture_imm_minus_fragmented"] - row["pooled_imm_minus_fragmented"]
        rows.append(row)
    return pd.DataFrame(rows)


def gate_summary(
    session_count: int,
    decoder: pd.DataFrame,
    selection: pd.DataFrame,
    evidence: pd.DataFrame,
    decisions: pd.DataFrame,
    direction: pd.DataFrame,
    max_events_per_session: int,
) -> pd.DataFrame:
    expected_events = int(session_count * max_events_per_session)
    successful = evidence[evidence["status"] == "success"] if "status" in evidence.columns else pd.DataFrame()
    models_per_event = successful.groupby(["session", "event_id", "encoding_variant"])["model"].nunique() if not successful.empty else pd.Series(dtype=int)
    checks = [
        ("native_ripple_sessions_present", session_count > 0, f"sessions={session_count}"),
        ("multiple_animals_present", selection["animal"].nunique() >= 2 if not selection.empty else False, f"animals={selection['animal'].nunique() if not selection.empty else 0}"),
        ("linear_and_circular_present", set(selection.get("geometry", [])) == {"linear", "circular"}, f"geometries={sorted(set(selection.get('geometry', [])))}"),
        ("all_sessions_have_decoder_output", len(decoder) == session_count and session_count > 0, f"decoder_rows={len(decoder)}/{session_count}"),
        ("at_least_one_decoder_pass_per_geometry", set(decoder.loc[decoder["decoder_qc_passed"].astype(bool), "geometry"]) == {"linear", "circular"} if not decoder.empty else False, "descriptive readiness gate"),
        ("balanced_event_target_complete", len(selection) == expected_events and expected_events > 0, f"selected={len(selection)}/{expected_events}"),
        ("required_models_complete", bool(len(models_per_event) > 0 and (models_per_event == len(MODELS)).all()), f"complete_groups={int((models_per_event == len(MODELS)).sum())}/{len(models_per_event)}"),
        (
            "no_model_scoring_failures",
            bool(not evidence.empty and "status" in evidence.columns and evidence["status"].eq("success").all()),
            f"failures={int((evidence['status'] != 'success').sum()) if not evidence.empty and 'status' in evidence.columns else 0}",
        ),
        ("claim_decisions_present", len(decisions) == len(selection) and len(selection) > 0, f"decisions={len(decisions)}/{len(selection)}"),
        ("direction_sensitivity_present", len(direction) == len(selection) and len(selection) > 0, f"rows={len(direction)}/{len(selection)}"),
    ]
    overall = all(passed for _, passed, _ in checks)
    checks.append(("overall_technical", overall, "biological outcomes are descriptive, not pass/fail"))
    return pd.DataFrame([{"gate": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks])


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dirs = discover_native_ripple_sessions(dataset_root)
    evidence_rows: list[dict[str, object]] = []
    decoder_rows: list[dict[str, object]] = []
    unit_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []

    for session_dir in session_dirs:
        animal = session_dir.parent.name
        session = session_dir.name
        track = load_track_samples(session_dir)
        spikes = load_spikes(session_dir)
        encodings, unit_qc = build_session_encodings(
            track,
            spikes,
            position_bin_size_cm=args.position_bin_size_cm,
            min_run_speed_cm_s=args.min_run_speed_cm_s,
            min_run_spikes=args.min_run_spikes,
            min_spatial_information=args.min_spatial_information,
            min_peak_rate_hz=args.min_peak_rate_hz,
            min_encoding_units=args.min_encoding_units,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
        )
        selected_units = encodings["pooled"][0].unit_ids
        unit_qc.insert(0, "session", session)
        unit_qc.insert(0, "animal", animal)
        unit_frames.append(unit_qc)
        decoder_metrics = decode_crossvalidated(
            track,
            spikes,
            selected_units,
            position_bin_size_cm=args.position_bin_size_cm,
            min_run_speed_cm_s=args.min_run_speed_cm_s,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            n_folds=args.decoder_folds,
            decode_window_s=args.decoder_window_s,
            max_decode_bins=args.decoder_max_bins,
        )
        decoder_rows.append(
            {
                "animal": animal,
                "session": session,
                "maze_type": track.maze_type,
                "geometry": track.topology,
                "track_length_cm": track.track_length_cm,
                "total_ca1_units": len(spikes.unit_ids),
                "encoding_units": len(selected_units),
                **decoder_metrics,
            }
        )
        events = load_native_post_nrem_events(
            session_dir,
            track,
            spikes,
            selected_units,
            min_event_spikes=args.min_event_spikes,
            min_event_active_units=args.min_event_active_units,
            max_events=args.max_events_per_session,
            event_ranking=args.event_ranking,
            selection_seed=args.selection_seed,
        )
        if events.empty:
            continue
        events.insert(0, "geometry", track.topology)
        events.insert(0, "maze_type", track.maze_type)
        events.insert(0, "session", session)
        events.insert(0, "animal", animal)
        selection_frames.append(events)
        for event in events.itertuples(index=False):
            score_start_s = max(0.0, float(event.start_time_s) - float(args.event_padding_s))
            score_end_s = float(event.end_time_s) + float(args.event_padding_s)
            edges = event_bin_edges(score_start_s, score_end_s, args.time_bin_s)
            counts = spike_count_matrix(spikes, selected_units, edges)
            scored_n_spikes = int(counts.sum())
            scored_n_active_units = int(np.sum(counts.sum(axis=0) > 0))
            for variant, maps in encodings.items():
                started = time.perf_counter()
                try:
                    scores = score_encoding_variant(
                        counts,
                        edges,
                        maps,
                        topology=track.topology,
                        track_length_cm=track.track_length_cm,
                        diffusion_sigma_cm_sqrt_s=args.diffusion_sigma_cm_sqrt_s,
                        stationary_sigma_cm=args.stationary_sigma_cm,
                        max_step_sigma=args.max_step_sigma,
                        imm_mode_stickiness=args.imm_mode_stickiness,
                    )
                    runtime = time.perf_counter() - started
                    for model, score in scores.items():
                        diagnostics = {
                            "mean_stationary_mode_probability": np.nan,
                            "mean_nonstationary_mode_probability": np.nan,
                            "fraction_time_map_nonstationary": np.nan,
                            "posterior_expected_path_length_cm": np.nan,
                            "posterior_net_displacement_cm": np.nan,
                            "posterior_path_speed_cm_s": np.nan,
                        }
                        if model == "first_order_imm":
                            diagnostics = imm_content_diagnostics(
                                np.asarray(score["posterior"]),
                                np.asarray(score["mode_posterior"]),
                                maps[0].bin_centers_cm,
                                track.topology,
                                track.track_length_cm,
                                float(event.end_time_s - event.start_time_s),
                            )
                        evidence_rows.append(
                            {
                                "animal": animal,
                                "session": session,
                                "maze_type": track.maze_type,
                                "geometry": track.topology,
                                "event_id": int(event.event_id),
                                "selection_rank_within_session": int(event.selection_rank_within_session),
                                "encoding_variant": variant,
                                "model": model,
                                "model_family": "nontrajectory" if model == "stationary" else "trajectory",
                                "log_evidence": float(score["log_evidence"]),
                                "status": "success",
                                "failure_reason": "",
                                "runtime_s": runtime / len(MODELS),
                                "duration_ms": float(1000.0 * (score_end_s - score_start_s)),
                                "raw_ripple_duration_ms": float(event.duration_ms),
                                "event_padding_ms_each_side": float(1000.0 * args.event_padding_s),
                                "n_spikes": scored_n_spikes,
                                "raw_ripple_n_spikes": int(event.n_spikes),
                                "n_active_units": scored_n_active_units,
                                "raw_ripple_n_active_units": int(event.n_active_units),
                                "n_time_bins": int(len(edges) - 1),
                                "track_length_cm": track.track_length_cm,
                                "n_encoding_units": len(selected_units),
                                "topology_transition": "periodic" if track.topology == "circular" else "reflecting_endpoints",
                                "evidence_support": "exact_full_grid",
                                "evidence_comparable": True,
                                **diagnostics,
                            }
                        )
                except Exception as exc:
                    runtime = time.perf_counter() - started
                    for model in MODELS:
                        evidence_rows.append(
                            {
                                "animal": animal,
                                "session": session,
                                "maze_type": track.maze_type,
                                "geometry": track.topology,
                                "event_id": int(event.event_id),
                                "selection_rank_within_session": int(event.selection_rank_within_session),
                                "encoding_variant": variant,
                                "model": model,
                                "model_family": "nontrajectory" if model == "stationary" else "trajectory",
                                "log_evidence": np.nan,
                                "status": "failure",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                                "runtime_s": runtime / len(MODELS),
                                "duration_ms": float(1000.0 * (score_end_s - score_start_s)),
                                "raw_ripple_duration_ms": float(event.duration_ms),
                                "event_padding_ms_each_side": float(1000.0 * args.event_padding_s),
                                "n_spikes": scored_n_spikes,
                                "raw_ripple_n_spikes": int(event.n_spikes),
                                "n_active_units": scored_n_active_units,
                                "raw_ripple_n_active_units": int(event.n_active_units),
                                "n_time_bins": int(len(edges) - 1),
                                "track_length_cm": track.track_length_cm,
                                "n_encoding_units": len(selected_units),
                                "topology_transition": "periodic" if track.topology == "circular" else "reflecting_endpoints",
                                "evidence_support": "exact_full_grid",
                                "evidence_comparable": True,
                            }
                        )

    evidence = pd.DataFrame(evidence_rows)
    decoder = pd.DataFrame(decoder_rows)
    units = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    selection = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    decisions = event_decisions(evidence, margin_threshold=args.margin_threshold) if not evidence.empty else pd.DataFrame()
    by_session = summarize_decisions(decisions, ["animal", "session", "geometry"]) if not decisions.empty else pd.DataFrame()
    by_animal = summarize_decisions(decisions, ["animal"]) if not decisions.empty else pd.DataFrame()
    by_geometry = summarize_decisions(decisions, ["geometry"]) if not decisions.empty else pd.DataFrame()
    direction = direction_sensitivity(evidence) if not evidence.empty else pd.DataFrame()
    gates = gate_summary(len(session_dirs), decoder, selection, evidence, decisions, direction, args.max_events_per_session)

    outputs = {
        EVIDENCE_OUTPUT: evidence,
        DECISION_OUTPUT: decisions,
        DECODER_OUTPUT: decoder,
        UNIT_OUTPUT: units,
        SELECTION_OUTPUT: selection,
        SESSION_OUTPUT: by_session,
        ANIMAL_OUTPUT: by_animal,
        GEOMETRY_OUTPUT: by_geometry,
        DIRECTION_OUTPUT: direction,
        GATE_OUTPUT: gates,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "hc-11_Grosmark_Buzsaki_Webshare",
        "event_definition": "native_ripplesNREM_intersect_POST_and_NREM_then_spike_support_ranked_by_peak_power",
        "primary_encoding_variant": PRIMARY_ENCODING_VARIANT,
        "models": list(MODELS),
        "claim_boundary": "native-ripple geometry pilot; no Gate 2/3/4 biological IMM claim",
        "parameters": {key: value for key, value in vars(args).items() if key not in {"dataset_root", "output_dir"}},
        "sessions": [path.name for path in session_dirs],
        "selected_events": int(len(selection)),
        "evidence_rows": int(len(evidence)),
        **build_script_provenance(input_paths={"dataset_root": dataset_root}),
    }
    (output_dir / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / SUMMARY_OUTPUT).write_text(build_markdown_summary(decisions, decoder, by_geometry, direction, gates), encoding="utf-8")
    return {
        "evidence": evidence,
        "decisions": decisions,
        "decoder": decoder,
        "units": units,
        "selection": selection,
        "by_session": by_session,
        "by_animal": by_animal,
        "by_geometry": by_geometry,
        "direction": direction,
        "gates": gates,
    }


def build_markdown_summary(
    decisions: pd.DataFrame,
    decoder: pd.DataFrame,
    by_geometry: pd.DataFrame,
    direction: pd.DataFrame,
    gates: pd.DataFrame,
) -> str:
    technical = bool(gates.loc[gates["gate"] == "overall_technical", "passed"].iloc[0]) if not gates.empty else False
    lines = [
        "# hc-11 native-ripple geometry pilot",
        "",
        f"Technical status: **{'pass' if technical else 'fail'}**.",
        "",
        "This run uses native POST-NREM ripple events, exact full-grid stationary/diffusion/fragmented/first-order-IMM rows, periodic circular-maze transitions, and reflecting linear-maze endpoints.",
        "Biological model outcomes are descriptive only until the external Gate 2/3/4 ladder is run.",
        "",
        "## Decoder QC",
        "",
    ]
    if decoder.empty:
        lines.append("No decoder rows were produced.")
    else:
        lines.extend(["```text", decoder.to_string(index=False), "```"])
    lines.extend(["", "## Geometry-stratified evidence", ""])
    lines.extend(["```text", by_geometry.to_string(index=False), "```"] if not by_geometry.empty else ["No model decisions were produced."])
    lines.extend(["", "## Direction sensitivity", ""])
    if direction.empty:
        lines.append("No direction-sensitivity rows were produced.")
    else:
        lines.extend(
            [
                f"Median change in trajectory-minus-stationary after direction conditioning: {direction['direction_conditioning_delta_trajectory_margin'].median():+.3f} log evidence.",
                f"Median change in IMM-minus-fragmented after direction conditioning: {direction['direction_conditioning_delta_imm_margin'].median():+.3f} log evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "hc-11 is a constrained linear/circular maze dataset, not a 2D open field. A positive result would test generalization across track topology; it would not by itself replicate unconstrained 2D geometry.",
            "The cohort has five native-ripple sessions from two animals, so animal-level generality remains limited even when event counts are large.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-events-per-session", type=int, default=20)
    parser.add_argument("--time-bin-s", type=float, default=0.010)
    parser.add_argument("--event-padding-s", type=float, default=0.0)
    parser.add_argument("--position-bin-size-cm", type=float, default=4.0)
    parser.add_argument("--min-run-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--min-run-spikes", type=int, default=20)
    parser.add_argument("--min-spatial-information", type=float, default=0.1)
    parser.add_argument("--min-peak-rate-hz", type=float, default=1.0)
    parser.add_argument("--min-encoding-units", type=int, default=5)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=1.5)
    parser.add_argument("--min-event-spikes", type=int, default=8)
    parser.add_argument("--min-event-active-units", type=int, default=4)
    parser.add_argument(
        "--event-ranking",
        choices=("peak_ripple_power", "spike_support", "random"),
        default="peak_ripple_power",
    )
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--decoder-folds", type=int, default=5)
    parser.add_argument("--decoder-window-s", type=float, default=0.250)
    parser.add_argument("--decoder-max-bins", type=int, default=2000)
    parser.add_argument("--diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--max-step-sigma", type=float, default=4.0)
    parser.add_argument("--imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
