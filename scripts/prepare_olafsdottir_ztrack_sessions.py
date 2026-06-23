#!/usr/bin/env python3
"""Build provisional Pfeiffer/Foster-style sessions from Olafsdottir Z-track data.

This is a bridge adapter, not a final native Olafsdottir data model.  It
converts Track1/SleepPOST Axona day pairs into the MAT layout already consumed
by the existing Pfeiffer/Foster benchmark scripts:

* Position_Data.mat
* Spike_Data.mat
* Ripple_Events.mat
* Epochs.mat
* Experiment_Information.mat

Track1 is used for encoding through ``Epochs.Run_Times``.  SleepPOST spikes and
candidate ripple events are shifted after the Track1 epoch so the current
session loader can score post-track replay windows without learning place fields
from SleepPOST.  The time shift is practical glue for the existing loader and is
recorded in the derived metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import heapq
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, hilbert, sosfiltfilt

from hipporeplayimm.olafsdottir2016 import (
    read_axona_cut,
    read_axona_egf,
    read_axona_pos,
    read_axona_set,
    tetrode_arrangement_for_animal,
)


ADAPTER_SCHEMA_VERSION = "olafsdottir_ztrack_pfeiffer_bridge_v1"
SOURCE_DATASET = "Olafsdottir2016"
DEFAULT_CACHE_ROOT = Path("/home/github-runner/.cache/datasets/olafsdottir2016")
DEFAULT_ARCHIVE = DEFAULT_CACHE_ROOT / "archive" / "Olafsdottir2016.zip"
DEFAULT_EXTRACTED_ROOT = DEFAULT_CACHE_ROOT / "extracted"
DEFAULT_DERIVED_ROOT = DEFAULT_CACHE_ROOT / "derived_pfeiffer"


@dataclass(frozen=True)
class DayPair:
    animal: str
    date: str
    day_dir: Path
    track_stem: str
    sleep_stem: str

    @property
    def rat_dir(self) -> str:
        return self.animal.upper()

    @property
    def session_name(self) -> str:
        return f"ZTrack{self.date.replace('-', '')}"

    @property
    def session_id(self) -> str:
        return f"{self.rat_dir}/{self.session_name}"


@dataclass(frozen=True)
class RippleDetectionConfig:
    channel_numbers: tuple[int, ...]
    band_low_hz: float
    band_high_hz: float
    high_threshold_z: float
    low_threshold_z: float
    min_duration_s: float
    max_duration_s: float
    expand_to_s: float
    exclude_sleep_onset_s: float
    detector_mode: str
    consensus_min_channels: int


@dataclass(frozen=True)
class ConversionConfig:
    tetrode_mode: str
    min_track_spikes: int
    min_sleep_spikes: int
    max_sleep_rate_hz: float
    min_event_spikes: int
    min_event_active_cells: int
    linear_position_bin_cm: float
    sleep_offset_padding_s: float
    ripple: RippleDetectionConfig


@dataclass(frozen=True)
class ParsedPosition:
    times_s: np.ndarray
    xy_cm: np.ndarray
    linear_cm: np.ndarray
    valid: np.ndarray
    pixels_per_metre: float


@dataclass(frozen=True)
class ParsedSpikes:
    spikes: np.ndarray
    tetrode_cell_ids: np.ndarray
    cell_summary: pd.DataFrame


def discover_day_pairs(root: str | Path) -> list[DayPair]:
    """Find extracted rat/day folders containing both Track1 and SleepPOST."""

    root_path = Path(root)
    pairs: list[DayPair] = []
    if not root_path.exists():
        return pairs
    for animal_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        if not animal_dir.name.lower().startswith("r"):
            continue
        for day_dir in sorted(path for path in animal_dir.iterdir() if path.is_dir()):
            set_files = sorted(day_dir.glob("*.set"))
            track = [path for path in set_files if "track1" in path.stem.lower()]
            sleep = [path for path in set_files if "sleeppost" in path.stem.lower()]
            if not track or not sleep:
                continue
            pairs.append(
                DayPair(
                    animal=animal_dir.name,
                    date=day_dir.name,
                    day_dir=day_dir,
                    track_stem=track[0].stem,
                    sleep_stem=sleep[0].stem,
                )
            )
    return pairs


def extract_day_from_zip(zip_path: str | Path, extracted_root: str | Path, animal: str, date: str) -> Path:
    """Extract a single rat/date folder from the Zenodo archive if missing."""

    root = Path(extracted_root)
    day_dir = root / animal.lower() / date
    if day_dir.exists() and any(day_dir.iterdir()):
        return day_dir
    prefix = f"{animal.lower()}/{date}/"
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().startswith(prefix)]
        if not members:
            raise FileNotFoundError(f"{zip_path} does not contain {prefix}")
        archive.extractall(root, members)
    return day_dir


def extract_all_day_pairs(zip_path: str | Path, extracted_root: str | Path) -> None:
    """Extract all day folders from the Zenodo archive."""

    root = Path(extracted_root)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)


def parse_position_file(path: str | Path, *, linear_bin_cm: float) -> ParsedPosition:
    """Read Track1 position and linearize it by occupied-bin geodesic distance."""

    pos = read_axona_pos(path)
    xy = np.column_stack([pos.x_cm, pos.y_cm])
    linear = linearize_position_geodesic(xy, bin_size_cm=linear_bin_cm)
    valid = pos.valid & np.isfinite(linear)
    return ParsedPosition(
        times_s=pos.times_s,
        xy_cm=xy,
        linear_cm=linear,
        valid=valid,
        pixels_per_metre=pos.pixels_per_metre,
    )


def linearize_position_geodesic(xy_cm: np.ndarray, *, bin_size_cm: float) -> np.ndarray:
    """Linearize a 2D occupied track by graph distance over occupied bins."""

    xy = np.asarray(xy_cm, dtype=float)
    out = np.full(xy.shape[0], np.nan, dtype=float)
    valid = np.isfinite(xy).all(axis=1)
    if valid.sum() < 2:
        return out

    scaled = np.floor(xy[valid] / float(bin_size_cm)).astype(int)
    cells, inverse = np.unique(scaled, axis=0, return_inverse=True)
    if cells.shape[0] < 2:
        out[valid] = 0.0
        return out

    keep_component = _largest_component(cells)
    component_indices = np.flatnonzero(keep_component)
    if component_indices.shape[0] < 2:
        out[valid] = 0.0
        return out

    component_cells = cells[component_indices]
    graph = _neighbor_graph(component_cells, bin_size_cm=bin_size_cm)
    endpoint_a = _farthest_node(graph, 0)[0]
    _endpoint_b, distances = _farthest_node(graph, endpoint_a)

    nearest_component = np.full(cells.shape[0], -1, dtype=int)
    nearest_component[component_indices] = np.arange(component_indices.shape[0])
    mapped = nearest_component[inverse]
    missing = mapped < 0
    if np.any(missing):
        for row in np.flatnonzero(missing):
            diffs = component_cells - scaled[row]
            mapped[row] = int(np.argmin(np.sum(diffs * diffs, axis=1)))
    linear = distances[mapped]
    linear -= np.nanmin(linear)
    out[valid] = linear
    return out


def _largest_component(cells: np.ndarray) -> np.ndarray:
    graph = _neighbor_graph(cells, bin_size_cm=1.0)
    visited = np.zeros(cells.shape[0], dtype=bool)
    best: list[int] = []
    for start in range(cells.shape[0]):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _weight in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        if len(component) > len(best):
            best = component
    keep = np.zeros(cells.shape[0], dtype=bool)
    keep[best] = True
    return keep


def _neighbor_graph(cells: np.ndarray, *, bin_size_cm: float) -> list[list[tuple[int, float]]]:
    by_coord = {tuple(map(int, coord)): index for index, coord in enumerate(cells)}
    graph: list[list[tuple[int, float]]] = [[] for _ in range(cells.shape[0])]
    for index, (x_coord, y_coord) in enumerate(cells):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = by_coord.get((int(x_coord + dx), int(y_coord + dy)))
                if neighbor is None:
                    continue
                graph[index].append((neighbor, math.hypot(dx, dy) * float(bin_size_cm)))
    return graph


def _farthest_node(graph: list[list[tuple[int, float]]], start: int) -> tuple[int, np.ndarray]:
    distances = np.full(len(graph), np.inf, dtype=float)
    distances[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    finite = np.isfinite(distances)
    if not finite.any():
        return start, distances
    farthest = int(np.flatnonzero(finite)[np.argmax(distances[finite])])
    return farthest, distances


def hippocampal_tetrodes(animal: str, mode: str) -> tuple[int, ...]:
    if mode == "all":
        return tuple(range(1, 17))
    if mode != "hippocampus":
        raise ValueError(f"unsupported tetrode mode: {mode}")
    return tetrode_arrangement_for_animal(animal).hippocampal_tetrodes


def parse_spikes_for_epoch(
    day_dir: str | Path,
    stem: str,
    tetrodes: tuple[int, ...],
    *,
    time_shift_s: float = 0.0,
) -> ParsedSpikes:
    rows: list[np.ndarray] = []
    cell_records: list[dict[str, object]] = []
    tetrode_cell_ids: list[tuple[int, int]] = []
    day = Path(day_dir)
    for tetrode in tetrodes:
        raw_path = day / f"{stem}.{tetrode}"
        cut_path = day / f"{stem}_{tetrode}.cut"
        if not raw_path.exists() or not cut_path.exists():
            continue
        try:
            cut = read_axona_cut(cut_path, tetrode_path=raw_path)
        except ValueError:
            continue
        if cut.spike_times_s is None:
            continue
        labels = np.asarray(cut.labels, dtype=int)
        times = np.asarray(cut.spike_times_s, dtype=float) + float(time_shift_s)
        for label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
            keep = labels == label
            if not np.any(keep):
                continue
            cell_id = int(tetrode * 100 + label)
            rows.append(np.column_stack([times[keep], np.full(int(keep.sum()), cell_id, dtype=float)]))
            tetrode_cell_ids.append((tetrode, cell_id))
            cell_records.append(
                {
                    "tetrode": tetrode,
                    "cluster": label,
                    "cell_id": cell_id,
                    "spikes": int(keep.sum()),
                    "first_spike_s": float(np.min(times[keep])),
                    "last_spike_s": float(np.max(times[keep])),
                }
            )
    spikes = np.vstack(rows) if rows else np.empty((0, 2), dtype=float)
    if spikes.size:
        spikes = spikes[np.argsort(spikes[:, 0], kind="mergesort")]
    return ParsedSpikes(
        spikes=spikes,
        tetrode_cell_ids=np.asarray(tetrode_cell_ids, dtype=int).reshape(-1, 2)
        if tetrode_cell_ids
        else np.empty((0, 2), dtype=int),
        cell_summary=pd.DataFrame(cell_records),
    )


def restrict_to_common_cells(
    track: ParsedSpikes,
    sleep: ParsedSpikes,
    *,
    min_track_spikes: int,
    min_sleep_spikes: int,
    max_sleep_rate_hz: float,
    sleep_duration_s: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    track_counts = _cell_counts(track.spikes)
    sleep_counts = _cell_counts(sleep.spikes)
    common = sorted(set(track_counts).intersection(sleep_counts))
    rows = []
    keep_ids: list[int] = []
    for cell_id in common:
        track_n = int(track_counts[cell_id])
        sleep_n = int(sleep_counts[cell_id])
        sleep_rate = sleep_n / max(float(sleep_duration_s), 1e-9)
        included = (
            track_n >= int(min_track_spikes)
            and sleep_n >= int(min_sleep_spikes)
            and sleep_rate <= float(max_sleep_rate_hz)
        )
        rows.append(
            {
                "cell_id": cell_id,
                "track_spikes": track_n,
                "sleep_spikes": sleep_n,
                "sleep_rate_hz": sleep_rate,
                "included": bool(included),
            }
        )
        if included:
            keep_ids.append(cell_id)
    keep = np.asarray(keep_ids, dtype=int)
    return _filter_spikes_by_cell(track.spikes, keep), _filter_spikes_by_cell(sleep.spikes, keep), pd.DataFrame(rows)


def _cell_counts(spikes: np.ndarray) -> dict[int, int]:
    if spikes.size == 0:
        return {}
    ids, counts = np.unique(spikes[:, 1].astype(int), return_counts=True)
    return {int(cell_id): int(count) for cell_id, count in zip(ids, counts)}


def _filter_spikes_by_cell(spikes: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    if spikes.size == 0 or cell_ids.size == 0:
        return np.empty((0, 2), dtype=float)
    keep = np.isin(spikes[:, 1].astype(int), cell_ids.astype(int))
    return np.asarray(spikes[keep], dtype=float)


def robust_z(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    median = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(arr, dtype=float)
    return (arr - median) / scale


def ripple_envelope_z(signal: np.ndarray, sample_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = 0.5 * float(sample_rate_hz)
    high = min(float(high_hz), 0.9 * nyquist)
    low = min(float(low_hz), 0.8 * high)
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, np.asarray(signal, dtype=float))
    envelope = np.abs(hilbert(filtered))
    return robust_z(envelope)


def detect_ripple_events_from_traces(
    traces_z: np.ndarray,
    sample_rate_hz: float,
    config: RippleDetectionConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Detect candidate events from per-channel ripple-envelope z traces."""

    traces = np.asarray(traces_z, dtype=float)
    if traces.ndim != 2:
        raise ValueError("traces_z must be channel x time")
    if traces.shape[0] == 0 or traces.shape[1] == 0:
        return np.empty((0, 6), dtype=float), pd.DataFrame()

    if config.detector_mode == "mean-envelope":
        score = np.nanmean(traces, axis=0)
        events = _events_from_crossing(
            score,
            score >= config.high_threshold_z,
            score >= config.low_threshold_z,
            sample_rate_hz,
            config,
        )
    elif config.detector_mode in {"per-channel-union", "per-channel-consensus"}:
        channel_events: list[list[float]] = []
        for trace in traces:
            channel_events.extend(
                _events_from_crossing(
                    trace,
                    trace >= config.high_threshold_z,
                    trace >= config.low_threshold_z,
                    sample_rate_hz,
                    config,
                ).tolist()
            )
        events = _merge_channel_events(np.asarray(channel_events, dtype=float), traces, sample_rate_hz, config)
    else:
        raise ValueError(f"unknown detector mode: {config.detector_mode}")

    if events.size == 0:
        return events, pd.DataFrame()
    channel_high = traces >= config.high_threshold_z
    peak_indices = np.clip(np.round(events[:, 2] * sample_rate_hz).astype(int), 0, traces.shape[1] - 1)
    event_table = pd.DataFrame(
        {
            "event_index": np.arange(events.shape[0]),
            "start_s": events[:, 0],
            "end_s": events[:, 1],
            "peak_s": events[:, 2],
            "peak_score_z": events[:, 3],
            "duration_s": events[:, 1] - events[:, 0],
            "channels_above_high_at_peak": np.sum(channel_high[:, peak_indices], axis=0),
            "detector_mode": config.detector_mode,
        }
    )
    return events, event_table


def _events_from_crossing(
    score: np.ndarray,
    crossing: np.ndarray,
    support: np.ndarray,
    sample_rate_hz: float,
    config: RippleDetectionConfig,
) -> np.ndarray:
    starts = np.flatnonzero(np.diff(np.r_[False, crossing, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, crossing, False].astype(int)) == -1)
    rows: list[list[float]] = []
    for start, end in zip(starts, ends):
        left = start
        while left > 0 and support[left - 1]:
            left -= 1
        right = end
        while right < support.shape[0] and support[right]:
            right += 1
        if right <= left:
            continue
        peak = int(left + np.nanargmax(score[left:right]))
        peak_s = peak / float(sample_rate_hz)
        if peak_s < config.exclude_sleep_onset_s:
            continue
        event_start = left / float(sample_rate_hz)
        event_end = right / float(sample_rate_hz)
        if config.expand_to_s > 0.0:
            half_width = 0.5 * float(config.expand_to_s)
            event_start = max(config.exclude_sleep_onset_s, peak_s - half_width)
            event_end = peak_s + half_width
        duration = event_end - event_start
        if duration < config.min_duration_s or duration > config.max_duration_s:
            continue
        peak_value = float(score[peak])
        rows.append([event_start, event_end, peak_s, peak_value, peak_value, peak_value])
    if not rows:
        return np.empty((0, 6), dtype=float)
    return _merge_overlapping_event_rows(np.asarray(rows, dtype=float))


def _merge_overlapping_event_rows(events: np.ndarray) -> np.ndarray:
    if events.size == 0:
        return np.empty((0, 6), dtype=float)
    ordered = events[np.argsort(events[:, 0])]
    merged: list[list[float]] = []
    for row in ordered:
        current = row.tolist()
        if not merged or current[0] > merged[-1][1]:
            merged.append(current)
            continue
        merged[-1][1] = max(merged[-1][1], current[1])
        if current[3] > merged[-1][3]:
            merged[-1][2:] = current[2:]
    return np.asarray(merged, dtype=float)


def _merge_channel_events(
    events: np.ndarray,
    traces_z: np.ndarray,
    sample_rate_hz: float,
    config: RippleDetectionConfig,
) -> np.ndarray:
    if events.size == 0:
        return np.empty((0, 6), dtype=float)
    ordered = events[np.argsort(events[:, 0])]
    merged: list[list[float]] = []
    for row in ordered:
        if not merged or float(row[0]) > merged[-1][1]:
            merged.append(row.tolist())
        else:
            merged[-1][1] = max(merged[-1][1], float(row[1]))
            if row[3] > merged[-1][3]:
                merged[-1][2:] = row[2:].tolist()
    out: list[list[float]] = []
    for row in merged:
        start_idx = max(0, int(math.floor(row[0] * sample_rate_hz)))
        end_idx = min(traces_z.shape[1], int(math.ceil(row[1] * sample_rate_hz)))
        if end_idx <= start_idx:
            continue
        peak_support = np.max(traces_z[:, start_idx:end_idx], axis=1) >= config.high_threshold_z
        if config.detector_mode == "per-channel-consensus" and int(peak_support.sum()) < config.consensus_min_channels:
            continue
        mean_score = np.nanmean(traces_z[:, start_idx:end_idx], axis=0)
        peak_local = int(np.nanargmax(mean_score))
        peak_idx = start_idx + peak_local
        peak_value = float(mean_score[peak_local])
        out.append([row[0], row[1], peak_idx / sample_rate_hz, peak_value, peak_value, peak_value])
    return np.asarray(out, dtype=float) if out else np.empty((0, 6), dtype=float)


def detect_ripples(day_dir: str | Path, sleep_stem: str, config: RippleDetectionConfig) -> tuple[np.ndarray, pd.DataFrame]:
    traces: list[np.ndarray] = []
    channel_rows = []
    sample_rate = None
    day = Path(day_dir)
    for channel in config.channel_numbers:
        path = _egf_path(day, sleep_stem, channel)
        if not path.exists():
            continue
        egf = read_axona_egf(path)
        if sample_rate is None:
            sample_rate = float(egf.sample_rate_hz)
        elif abs(sample_rate - float(egf.sample_rate_hz)) > 1e-6:
            raise ValueError(f"EGF sample rate mismatch: {sample_rate} vs {egf.sample_rate_hz}")
        z = ripple_envelope_z(egf.signal.astype(float), egf.sample_rate_hz, config.band_low_hz, config.band_high_hz)
        traces.append(z)
        channel_rows.append(
            {
                "channel": channel,
                "path": path.name,
                "sample_rate_hz": float(egf.sample_rate_hz),
                "samples": int(egf.signal.shape[0]),
                "max_ripple_z": float(np.nanmax(z)),
                "p99_ripple_z": float(np.nanpercentile(z, 99.0)),
                "p999_ripple_z": float(np.nanpercentile(z, 99.9)),
            }
        )
    if not traces or sample_rate is None:
        return np.empty((0, 6), dtype=float), pd.DataFrame(channel_rows)
    min_len = min(trace.shape[0] for trace in traces)
    matrix = np.vstack([trace[:min_len] for trace in traces])
    events, event_table = detect_ripple_events_from_traces(matrix, sample_rate, config)
    channel_summary = pd.DataFrame(channel_rows)
    if not event_table.empty:
        event_table["detector_mode"] = config.detector_mode
    return events, channel_summary


def event_spike_quality(ripple_events: np.ndarray, sleep_spikes_shifted: np.ndarray) -> pd.DataFrame:
    rows = []
    events = np.asarray(ripple_events, dtype=float).reshape(-1, 6)
    for event_index, event in enumerate(events):
        start, end = float(event[0]), float(event[1])
        if sleep_spikes_shifted.size:
            in_event = (sleep_spikes_shifted[:, 0] >= start) & (sleep_spikes_shifted[:, 0] <= end)
            event_spikes = sleep_spikes_shifted[in_event]
        else:
            event_spikes = np.empty((0, 2), dtype=float)
        rows.append(
            {
                "event_index": event_index,
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
                "n_spikes": int(event_spikes.shape[0]),
                "active_cell_count": int(np.unique(event_spikes[:, 1]).shape[0]) if event_spikes.size else 0,
            }
        )
    return pd.DataFrame(rows)


def filter_events_by_spike_support(
    ripple_events: np.ndarray,
    quality: pd.DataFrame,
    *,
    min_event_spikes: int,
    min_event_active_cells: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if quality.empty or ripple_events.size == 0:
        return ripple_events.reshape(-1, 6), quality
    keep = (quality["n_spikes"] >= int(min_event_spikes)) & (
        quality["active_cell_count"] >= int(min_event_active_cells)
    )
    filtered_events = np.asarray(ripple_events, dtype=float).reshape(-1, 6)[keep.to_numpy()]
    filtered_quality = quality.loc[keep].copy().reset_index(drop=True)
    filtered_quality["event_index"] = np.arange(len(filtered_quality), dtype=int)
    return filtered_events, filtered_quality


def build_session(day: DayPair, output_root: str | Path, config: ConversionConfig) -> dict[str, object]:
    """Convert one Track1/SleepPOST day pair into a derived MAT session."""

    tetrodes = hippocampal_tetrodes(day.animal, config.tetrode_mode)
    track_position = parse_position_file(day.day_dir / f"{day.track_stem}.pos", linear_bin_cm=config.linear_position_bin_cm)
    sleep_header = read_axona_set(day.day_dir / f"{day.sleep_stem}.set")
    track_header = read_axona_set(day.day_dir / f"{day.track_stem}.set")
    sleep_duration_s = _header_float(sleep_header, "duration", 0.0)
    track_duration_s = _header_float(track_header, "duration", float(np.nanmax(track_position.times_s)))
    position_end = float(np.nanmax(track_position.times_s)) if track_position.times_s.size else 0.0
    sleep_offset = max(track_duration_s + config.sleep_offset_padding_s, position_end + config.sleep_offset_padding_s)

    track_spikes_all = parse_spikes_for_epoch(day.day_dir, day.track_stem, tetrodes, time_shift_s=0.0)
    sleep_spikes_all = parse_spikes_for_epoch(day.day_dir, day.sleep_stem, tetrodes, time_shift_s=sleep_offset)
    track_spikes, sleep_spikes, cell_quality = restrict_to_common_cells(
        track_spikes_all,
        sleep_spikes_all,
        min_track_spikes=config.min_track_spikes,
        min_sleep_spikes=config.min_sleep_spikes,
        max_sleep_rate_hz=config.max_sleep_rate_hz,
        sleep_duration_s=sleep_duration_s,
    )

    combined_spikes = (
        np.vstack([track_spikes, sleep_spikes])
        if track_spikes.size or sleep_spikes.size
        else np.empty((0, 2), dtype=float)
    )
    if combined_spikes.size:
        combined_spikes = combined_spikes[np.argsort(combined_spikes[:, 0], kind="mergesort")]
    cell_ids = np.asarray(sorted(np.unique(combined_spikes[:, 1].astype(int)))) if combined_spikes.size else np.empty(0, dtype=int)
    tetrode_cell_ids = np.asarray([[cell_id // 100, cell_id] for cell_id in cell_ids], dtype=int)

    ripple_events_sleep, ripple_channel_summary = detect_ripples(day.day_dir, day.sleep_stem, config.ripple)
    ripple_events = ripple_events_sleep.copy()
    if ripple_events.size:
        ripple_events[:, :3] += sleep_offset
    unfiltered_quality = event_spike_quality(ripple_events, sleep_spikes)
    ripple_events, quality = filter_events_by_spike_support(
        ripple_events,
        unfiltered_quality,
        min_event_spikes=config.min_event_spikes,
        min_event_active_cells=config.min_event_active_cells,
    )

    finite = track_position.valid & np.isfinite(track_position.linear_cm)
    position = np.column_stack(
        [
            track_position.times_s[finite],
            track_position.linear_cm[finite],
            np.zeros(int(finite.sum()), dtype=float),
        ]
    )

    session_dir = Path(output_root) / day.rat_dir / day.session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_mat_files(
        session_dir=session_dir,
        day=day,
        position=position,
        combined_spikes=combined_spikes,
        cell_ids=cell_ids,
        tetrode_cell_ids=tetrode_cell_ids,
        ripple_events=ripple_events,
        sleep_offset=sleep_offset,
        sleep_duration_s=sleep_duration_s,
        track_duration_s=track_duration_s,
        config=config,
    )

    cell_quality.to_csv(session_dir / "olafsdottir_cell_quality.csv", index=False)
    quality.to_csv(session_dir / "olafsdottir_ripple_event_quality.csv", index=False)
    unfiltered_quality.to_csv(session_dir / "olafsdottir_ripple_event_quality_unfiltered.csv", index=False)
    ripple_channel_summary.to_csv(session_dir / "olafsdottir_ripple_channel_summary.csv", index=False)

    summary = conversion_summary(
        day=day,
        session_dir=session_dir,
        position=position,
        track_duration_s=track_duration_s,
        sleep_duration_s=sleep_duration_s,
        sleep_offset=sleep_offset,
        tetrodes=tetrodes,
        cell_ids=cell_ids,
        combined_spikes=combined_spikes,
        track_spikes=track_spikes,
        sleep_spikes=sleep_spikes,
        ripple_events=ripple_events,
        unfiltered_quality=unfiltered_quality,
        quality=quality,
        config=config,
    )
    (session_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_mat_files(
    *,
    session_dir: Path,
    day: DayPair,
    position: np.ndarray,
    combined_spikes: np.ndarray,
    cell_ids: np.ndarray,
    tetrode_cell_ids: np.ndarray,
    ripple_events: np.ndarray,
    sleep_offset: float,
    sleep_duration_s: float,
    track_duration_s: float,
    config: ConversionConfig,
) -> None:
    event_filter_parameters = json.dumps(_event_filter_parameters(config), sort_keys=True, separators=(",", ":"))
    sio.savemat(session_dir / "Position_Data.mat", {"Position_Data": position})
    sio.savemat(
        session_dir / "Spike_Data.mat",
        {
            "Spike_Data": combined_spikes,
            "Tetrode_Cell_IDs": tetrode_cell_ids,
            "Excitatory_Neurons": cell_ids,
            "Inhibitory_Neurons": np.empty((0,), dtype=int),
        },
    )
    sio.savemat(session_dir / "Ripple_Events.mat", {"Ripple_Events": ripple_events.reshape(-1, 6)})
    run_times = (
        np.asarray([[float(np.nanmin(position[:, 0])), float(np.nanmax(position[:, 0]))]])
        if position.size
        else np.empty((0, 2), dtype=float)
    )
    sleep_times = np.asarray(
        [[sleep_offset + config.ripple.exclude_sleep_onset_s, sleep_offset + sleep_duration_s]],
        dtype=float,
    )
    sio.savemat(
        session_dir / "Epochs.mat",
        {
            "Run_Times": run_times,
            "Sleep_Times": sleep_times,
            "Sleep_Box_Immobile_Times": sleep_times,
            "REM_Times": np.empty((0, 2), dtype=float),
        },
    )
    sio.savemat(
        session_dir / "Experiment_Information.mat",
        {
            "source_dataset": SOURCE_DATASET,
            "source_animal": day.rat_dir,
            "source_track_session": day.track_stem,
            "source_sleep_session": day.sleep_stem,
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "linearization_method": "occupied_bin_geodesic",
            "event_detector": config.ripple.detector_mode,
            "event_filter_parameters": event_filter_parameters,
            "sleep_time_offset_s": float(sleep_offset),
            "track_duration_s": float(track_duration_s),
            "bridge_adapter_note": (
                "SleepPOST spikes and ripple events are shifted after Track1 so "
                "existing Pfeiffer/Foster-style loaders can fit encoding on Track1."
            ),
        },
    )


def conversion_summary(
    *,
    day: DayPair,
    session_dir: Path,
    position: np.ndarray,
    track_duration_s: float,
    sleep_duration_s: float,
    sleep_offset: float,
    tetrodes: tuple[int, ...],
    cell_ids: np.ndarray,
    combined_spikes: np.ndarray,
    track_spikes: np.ndarray,
    sleep_spikes: np.ndarray,
    ripple_events: np.ndarray,
    unfiltered_quality: pd.DataFrame,
    quality: pd.DataFrame,
    config: ConversionConfig,
) -> dict[str, object]:
    return {
        "session": day.session_id,
        "session_dir": str(session_dir),
        "source_dataset": SOURCE_DATASET,
        "source_animal": day.rat_dir,
        "source_track_session": day.track_stem,
        "source_sleep_session": day.sleep_stem,
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "linearization_method": "occupied_bin_geodesic",
        "event_detector": config.ripple.detector_mode,
        "event_filter_parameters": _event_filter_parameters(config),
        "sleep_time_offset_s": float(sleep_offset),
        "track_duration_s": float(track_duration_s),
        "sleep_duration_s": float(sleep_duration_s),
        "track_position_samples": int(position.shape[0]),
        "tetrode_mode": config.tetrode_mode,
        "tetrodes": list(tetrodes),
        "included_cells": int(cell_ids.shape[0]),
        "combined_spikes": int(combined_spikes.shape[0]),
        "track_spikes": int(track_spikes.shape[0]),
        "sleep_spikes": int(sleep_spikes.shape[0]),
        "ripple_events_unfiltered": int(len(unfiltered_quality)),
        "ripple_events": int(ripple_events.shape[0]),
        "median_event_spikes": float(quality["n_spikes"].median()) if not quality.empty else 0.0,
        "max_event_spikes": int(quality["n_spikes"].max()) if not quality.empty else 0,
        "ripple_detector": asdict(config.ripple),
        "conversion_config": {
            "min_track_spikes": config.min_track_spikes,
            "min_sleep_spikes": config.min_sleep_spikes,
            "max_sleep_rate_hz": config.max_sleep_rate_hz,
            "linear_position_bin_cm": config.linear_position_bin_cm,
            "sleep_offset_padding_s": config.sleep_offset_padding_s,
        },
    }


def _event_filter_parameters(config: ConversionConfig) -> dict[str, object]:
    return {
        "min_event_spikes": int(config.min_event_spikes),
        "min_event_active_cells": int(config.min_event_active_cells),
        "ripple": asdict(config.ripple),
    }


def _header_float(header: dict[str, str], key: str, default: float) -> float:
    raw = header.get(key)
    if raw is None:
        return float(default)
    import re

    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(raw))
    return float(match.group(0)) if match else float(default)


def _egf_path(day_dir: Path, stem: str, channel: int) -> Path:
    if int(channel) == 1:
        return day_dir / f"{stem}.egf"
    return day_dir / f"{stem}.egf{int(channel)}"


def _parse_channels(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = [int(value) for value in item.split("-", 1)]
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("at least one LFP channel is required")
    return tuple(sorted(dict.fromkeys(values)))


def _select_days(pairs: list[DayPair], sessions: str, max_days: int | None) -> list[DayPair]:
    selected = pairs
    if sessions.strip().lower() != "all":
        wanted = {item.strip().lower() for item in sessions.split(",") if item.strip()}
        selected = [
            pair
            for pair in pairs
            if pair.session_id.lower() in wanted or f"{pair.animal}/{pair.date}".lower() in wanted
        ]
    if max_days is not None:
        selected = selected[: int(max_days)]
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--extract", action="store_true", help="Extract the Zenodo archive before conversion.")
    parser.add_argument(
        "--sessions",
        default="all",
        help="Comma list of R2142/ZTrackYYYYMMDD or r2142/YYYY-MM-DD; default all discovered.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--tetrode-mode", choices=("hippocampus", "all"), default="hippocampus")
    parser.add_argument("--min-track-spikes", type=int, default=20)
    parser.add_argument("--min-sleep-spikes", type=int, default=5)
    parser.add_argument("--max-sleep-rate-hz", type=float, default=10.0)
    parser.add_argument("--min-event-spikes", type=int, default=0)
    parser.add_argument("--min-event-active-cells", type=int, default=0)
    parser.add_argument("--linear-position-bin-cm", type=float, default=3.0)
    parser.add_argument("--sleep-offset-padding-s", type=float, default=1000.0)
    parser.add_argument("--lfp-channels", default="1-4")
    parser.add_argument("--ripple-band-low-hz", type=float, default=150.0)
    parser.add_argument("--ripple-band-high-hz", type=float, default=250.0)
    parser.add_argument("--ripple-high-threshold-z", type=float, default=2.25)
    parser.add_argument("--ripple-low-threshold-z", type=float, default=0.75)
    parser.add_argument("--ripple-min-duration-s", type=float, default=0.025)
    parser.add_argument("--ripple-max-duration-s", type=float, default=0.250)
    parser.add_argument("--ripple-expand-to-s", type=float, default=0.150)
    parser.add_argument("--exclude-sleep-onset-s", type=float, default=10.0)
    parser.add_argument(
        "--lfp-detector-mode",
        choices=("mean-envelope", "per-channel-union", "per-channel-consensus"),
        default="mean-envelope",
        help="Filter/envelope each channel before combining z-scored envelopes.",
    )
    parser.add_argument("--consensus-min-channels", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.extract:
        extract_all_day_pairs(args.zip_path, args.extracted_root)

    pairs = discover_day_pairs(args.extracted_root)
    selected = _select_days(pairs, args.sessions, args.max_days)
    if not selected:
        raise RuntimeError(f"No Track1/SleepPOST day pairs found under {args.extracted_root}")

    ripple_config = RippleDetectionConfig(
        channel_numbers=_parse_channels(args.lfp_channels),
        band_low_hz=args.ripple_band_low_hz,
        band_high_hz=args.ripple_band_high_hz,
        high_threshold_z=args.ripple_high_threshold_z,
        low_threshold_z=args.ripple_low_threshold_z,
        min_duration_s=args.ripple_min_duration_s,
        max_duration_s=args.ripple_max_duration_s,
        expand_to_s=args.ripple_expand_to_s,
        exclude_sleep_onset_s=args.exclude_sleep_onset_s,
        detector_mode=args.lfp_detector_mode,
        consensus_min_channels=args.consensus_min_channels,
    )
    config = ConversionConfig(
        tetrode_mode=args.tetrode_mode,
        min_track_spikes=args.min_track_spikes,
        min_sleep_spikes=args.min_sleep_spikes,
        max_sleep_rate_hz=args.max_sleep_rate_hz,
        min_event_spikes=args.min_event_spikes,
        min_event_active_cells=args.min_event_active_cells,
        linear_position_bin_cm=args.linear_position_bin_cm,
        sleep_offset_padding_s=args.sleep_offset_padding_s,
        ripple=ripple_config,
    )
    summaries = [build_session(day, args.output_root, config) for day in selected]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_root / "olafsdottir_ztrack_conversion_summary.csv", index=False)
    (output_root / "olafsdottir_ztrack_conversion_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
