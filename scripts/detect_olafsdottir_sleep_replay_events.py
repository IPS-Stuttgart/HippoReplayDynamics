#!/usr/bin/env python3
"""Detect conservative SleepPOST replay/SWR candidate events for Olafsdottir 2016."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from hipporeplayimm.olafsdottir2016 import (
    MANIFEST_NAME,
    read_axona_cut,
    read_axona_egf,
)


EVENT_COLUMNS = [
    "event_index",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "peak_time_s",
    "peak_ripple_z",
    "n_spikes",
    "n_active_cells",
    "animal",
    "date",
    "track_session",
    "sleep_session",
    "event_detector",
    "detector_parameters",
]

SUMMARY_COLUMNS = [
    "animal",
    "date",
    "track_session",
    "sleep_session",
    "event_detector",
    "detector_parameters",
    "n_lfp_channels",
    "lfp_channel_paths",
    "n_spike_cells",
    "n_threshold_crossings",
    "n_duration_gate_events",
    "n_spike_supported_events",
    "n_active_cell_supported_events",
    "n_events",
    "median_event_spikes",
    "max_event_spikes",
    "median_peak_ripple_z",
    "max_peak_ripple_z",
    "caveat",
]


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {
        "animal",
        "date",
        "session_type",
        "session_name",
        "session_path",
        "hippocampal_tetrodes",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"manifest is missing required columns: {missing}")
    return manifest


def paired_track_sleep_sessions(
    manifest: pd.DataFrame,
    *,
    animal: str | None = None,
    date: str | None = None,
) -> list[tuple[pd.Series, pd.Series]]:
    frame = manifest.copy()
    if animal:
        frame = frame[frame["animal"].astype(str).str.upper().eq(animal.upper())]
    if date:
        frame = frame[frame["date"].astype(str).eq(str(date))]

    pairs: list[tuple[pd.Series, pd.Series]] = []
    for (_animal, _date), group in frame.groupby(["animal", "date"], sort=True):
        tracks = group[group["session_type"].astype(str).eq("track1")].sort_values("session_name")
        sleeps = group[group["session_type"].astype(str).eq("sleepPOST")].sort_values("session_name")
        if tracks.empty or sleeps.empty:
            continue
        pairs.append((tracks.iloc[0], sleeps.iloc[0]))
    return pairs


def detect_session_events(
    track_row: pd.Series,
    sleep_row: pd.Series,
    *,
    ripple_low_hz: float = 150.0,
    ripple_high_hz: float = 250.0,
    ripple_z_threshold: float = 3.2,
    min_duration_s: float = 0.005,
    max_duration_s: float = 0.5,
    merge_gap_s: float = 0.015,
    envelope_smooth_s: float = 0.008,
    min_event_spikes: int = 5,
    min_event_active_cells: int = 3,
    max_lfp_channels: int = 8,
    combine_method: str = "max",
    event_detector: str = "ripple_band_lfp_threshold_v1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detector_parameters = {
        "ripple_low_hz": float(ripple_low_hz),
        "ripple_high_hz": float(ripple_high_hz),
        "ripple_z_threshold": float(ripple_z_threshold),
        "min_duration_s": float(min_duration_s),
        "max_duration_s": float(max_duration_s),
        "merge_gap_s": float(merge_gap_s),
        "envelope_smooth_s": float(envelope_smooth_s),
        "min_event_spikes": int(min_event_spikes),
        "min_event_active_cells": int(min_event_active_cells),
        "max_lfp_channels": int(max_lfp_channels),
        "combine_method": str(combine_method),
    }
    detector_json = json.dumps(detector_parameters, sort_keys=True, separators=(",", ":"))
    animal = str(sleep_row["animal"])
    date = str(sleep_row["date"])
    track_session = str(track_row["session_name"])
    sleep_session = str(sleep_row["session_name"])
    sleep_stem = Path(str(sleep_row["session_path"]))
    hpc_tetrodes = _parse_tetrodes(str(sleep_row["hippocampal_tetrodes"]))

    lfp = load_sleep_lfp(
        sleep_stem,
        hpc_tetrodes,
        max_channels=max_lfp_channels,
    )
    if lfp.signals.size == 0:
        summary = _summary_row(
            animal=animal,
            date=date,
            track_session=track_session,
            sleep_session=sleep_session,
            event_detector=event_detector,
            detector_parameters=detector_json,
            n_lfp_channels=0,
            lfp_channel_paths="",
            n_spike_cells=0,
            caveat="no hippocampal EGF channels found",
        )
        return pd.DataFrame(columns=EVENT_COLUMNS), pd.DataFrame([summary], columns=SUMMARY_COLUMNS)

    ripple_z = combined_ripple_envelope_z(
        lfp.signals,
        sample_rate_hz=lfp.sample_rate_hz,
        ripple_low_hz=ripple_low_hz,
        ripple_high_hz=ripple_high_hz,
        envelope_smooth_s=envelope_smooth_s,
        combine_method=combine_method,
    )
    raw_windows = threshold_windows(
        ripple_z,
        lfp.times_s,
        threshold=float(ripple_z_threshold),
        merge_gap_s=float(merge_gap_s),
    )
    duration_windows = [
        window
        for window in raw_windows
        if float(min_duration_s) <= window["duration_s"] <= float(max_duration_s)
    ]
    cells = load_sleep_spike_cells(sleep_stem, hpc_tetrodes)
    event_rows: list[dict[str, object]] = []
    spike_supported = 0
    active_supported = 0
    for window in duration_windows:
        n_spikes, n_active_cells = spike_support_counts(
            cells,
            start_time_s=float(window["start_time_s"]),
            end_time_s=float(window["end_time_s"]),
        )
        if n_spikes >= int(min_event_spikes):
            spike_supported += 1
        if n_active_cells >= int(min_event_active_cells):
            active_supported += 1
        if n_spikes < int(min_event_spikes) or n_active_cells < int(min_event_active_cells):
            continue
        event_rows.append(
            {
                "event_index": len(event_rows),
                "start_time_s": float(window["start_time_s"]),
                "end_time_s": float(window["end_time_s"]),
                "duration_s": float(window["duration_s"]),
                "peak_time_s": float(window["peak_time_s"]),
                "peak_ripple_z": float(window["peak_ripple_z"]),
                "n_spikes": int(n_spikes),
                "n_active_cells": int(n_active_cells),
                "animal": animal,
                "date": date,
                "track_session": track_session,
                "sleep_session": sleep_session,
                "event_detector": event_detector,
                "detector_parameters": detector_json,
            }
        )
    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    summary = _summary_row(
        animal=animal,
        date=date,
        track_session=track_session,
        sleep_session=sleep_session,
        event_detector=event_detector,
        detector_parameters=detector_json,
        n_lfp_channels=len(lfp.channel_paths),
        lfp_channel_paths=" ".join(str(path) for path in lfp.channel_paths),
        n_spike_cells=len(cells),
        n_threshold_crossings=len(raw_windows),
        n_duration_gate_events=len(duration_windows),
        n_spike_supported_events=spike_supported,
        n_active_cell_supported_events=active_supported,
        n_events=len(events),
        median_event_spikes=float(events["n_spikes"].median()) if not events.empty else np.nan,
        max_event_spikes=int(events["n_spikes"].max()) if not events.empty else 0,
        median_peak_ripple_z=float(events["peak_ripple_z"].median()) if not events.empty else np.nan,
        max_peak_ripple_z=float(events["peak_ripple_z"].max()) if not events.empty else np.nan,
        caveat="candidate detector; review LFP/ripple diagnostics before treating as final SWR detection",
    )
    return events, pd.DataFrame([summary], columns=SUMMARY_COLUMNS)


class LfpBundle:
    def __init__(
        self,
        *,
        signals: np.ndarray,
        times_s: np.ndarray,
        sample_rate_hz: float,
        channel_paths: list[Path],
    ) -> None:
        self.signals = signals
        self.times_s = times_s
        self.sample_rate_hz = sample_rate_hz
        self.channel_paths = channel_paths


def load_sleep_lfp(
    sleep_stem: Path,
    hippocampal_tetrodes: list[int],
    *,
    max_channels: int = 8,
) -> LfpBundle:
    signals: list[np.ndarray] = []
    channel_paths: list[Path] = []
    sample_rate_hz: float | None = None
    times_s: np.ndarray | None = None
    for tetrode in hippocampal_tetrodes:
        path = _egf_path_for_channel(sleep_stem, tetrode)
        if not path.exists():
            continue
        egf = read_axona_egf(path)
        signals.append(egf.signal.astype(float))
        channel_paths.append(path)
        sample_rate_hz = float(egf.sample_rate_hz) if sample_rate_hz is None else sample_rate_hz
        times_s = egf.times_s if times_s is None else times_s
        if len(signals) >= int(max_channels):
            break
    if not signals:
        return LfpBundle(
            signals=np.empty((0, 0), dtype=float),
            times_s=np.empty(0, dtype=float),
            sample_rate_hz=np.nan,
            channel_paths=[],
        )
    length = min(signal.shape[0] for signal in signals)
    return LfpBundle(
        signals=np.vstack([signal[:length] for signal in signals]),
        times_s=np.asarray(times_s[:length], dtype=float),
        sample_rate_hz=float(sample_rate_hz),
        channel_paths=channel_paths,
    )


def combined_ripple_envelope_z(
    signals: np.ndarray,
    *,
    sample_rate_hz: float,
    ripple_low_hz: float,
    ripple_high_hz: float,
    envelope_smooth_s: float,
    combine_method: str = "mean",
) -> np.ndarray:
    arr = np.asarray(signals, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.empty(0, dtype=float)
    high = min(float(ripple_high_hz), 0.45 * float(sample_rate_hz))
    low = min(float(ripple_low_hz), high * 0.8)
    sos = butter(3, [low, high], btype="bandpass", fs=float(sample_rate_hz), output="sos")
    z_channels = []
    for signal in arr:
        filtered = sosfiltfilt(sos, signal)
        envelope = np.abs(hilbert(filtered))
        envelope = _moving_average(envelope, max(1, int(round(float(envelope_smooth_s) * float(sample_rate_hz)))))
        z_channels.append(_robust_z(envelope))
    z = np.vstack(z_channels)
    if combine_method == "max":
        return np.nanmax(z, axis=0)
    if combine_method != "mean":
        raise ValueError("combine_method must be 'mean' or 'max'")
    return np.nanmean(z, axis=0)


def threshold_windows(
    z: np.ndarray,
    times_s: np.ndarray,
    *,
    threshold: float,
    merge_gap_s: float,
) -> list[dict[str, float]]:
    above = np.asarray(z, dtype=float) >= float(threshold)
    times = np.asarray(times_s, dtype=float)
    if above.size == 0:
        return []
    changes = np.diff(above.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    windows = [_window_from_indices(z, times, int(start), int(end)) for start, end in zip(starts, ends)]
    if not windows:
        return []
    merged = [windows[0]]
    for window in windows[1:]:
        if window["start_time_s"] - merged[-1]["end_time_s"] <= float(merge_gap_s):
            start_idx = int(merged[-1]["start_index"])
            end_idx = int(window["end_index"])
            merged[-1] = _window_from_indices(z, times, start_idx, end_idx)
        else:
            merged.append(window)
    return merged


def load_sleep_spike_cells(sleep_stem: Path, hippocampal_tetrodes: list[int]) -> list[tuple[str, np.ndarray]]:
    cells: list[tuple[str, np.ndarray]] = []
    for tetrode in hippocampal_tetrodes:
        cut_path = _cut_path_for_tetrode(sleep_stem, tetrode)
        tetrode_path = sleep_stem.with_suffix(f".{tetrode}")
        if cut_path is None or not tetrode_path.exists():
            continue
        try:
            cut = read_axona_cut(cut_path, tetrode_path=tetrode_path)
        except ValueError:
            continue
        if cut.spike_times_s is None:
            continue
        labels = np.asarray(cut.labels, dtype=int)
        times = np.asarray(cut.spike_times_s, dtype=float)
        for label in sorted(set(labels.tolist())):
            if label <= 0:
                continue
            cells.append((f"t{tetrode}:c{label}", times[labels == label]))
    return cells


def spike_support_counts(
    cells: list[tuple[str, np.ndarray]],
    *,
    start_time_s: float,
    end_time_s: float,
) -> tuple[int, int]:
    n_spikes = 0
    n_active_cells = 0
    for _cell_id, times in cells:
        count = int(np.count_nonzero((times >= start_time_s) & (times <= end_time_s)))
        n_spikes += count
        if count > 0:
            n_active_cells += 1
    return n_spikes, n_active_cells


def detect_all_sessions(
    manifest: pd.DataFrame,
    *,
    animal: str | None = None,
    date: str | None = None,
    **kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for track_row, sleep_row in paired_track_sleep_sessions(manifest, animal=animal, date=date):
        session_events, session_summary = detect_session_events(track_row, sleep_row, **kwargs)
        events.append(session_events)
        summaries.append(session_summary)
    event_table = pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=EVENT_COLUMNS)
    if not event_table.empty:
        event_table["event_index"] = np.arange(len(event_table), dtype=int)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=SUMMARY_COLUMNS)
    return event_table, summary


def write_detection_outputs(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    *,
    animal: str | None = None,
    date: str | None = None,
    **kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, summary = detect_all_sessions(manifest, animal=animal, date=date, **kwargs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    events.to_csv(out / "sleep_replay_events.csv", index=False)
    summary.to_csv(out / "ripple_detection_summary.csv", index=False)
    return events, summary


def _window_from_indices(z: np.ndarray, times_s: np.ndarray, start: int, end: int) -> dict[str, float]:
    segment = np.asarray(z[start:end], dtype=float)
    peak_offset = int(np.nanargmax(segment)) if segment.size else 0
    peak_index = min(start + peak_offset, max(start, end - 1))
    end_index = max(start, end - 1)
    return {
        "start_index": int(start),
        "end_index": int(end),
        "start_time_s": float(times_s[start]),
        "end_time_s": float(times_s[end_index]),
        "duration_s": float(times_s[end_index] - times_s[start]),
        "peak_time_s": float(times_s[peak_index]),
        "peak_ripple_z": float(z[peak_index]),
    }


def _summary_row(**kwargs: object) -> dict[str, object]:
    row: dict[str, object] = {
        "animal": "",
        "date": "",
        "track_session": "",
        "sleep_session": "",
        "event_detector": "ripple_band_lfp_threshold_v1",
        "detector_parameters": "{}",
        "n_lfp_channels": 0,
        "lfp_channel_paths": "",
        "n_spike_cells": 0,
        "n_threshold_crossings": 0,
        "n_duration_gate_events": 0,
        "n_spike_supported_events": 0,
        "n_active_cell_supported_events": 0,
        "n_events": 0,
        "median_event_spikes": np.nan,
        "max_event_spikes": 0,
        "median_peak_ripple_z": np.nan,
        "max_peak_ripple_z": np.nan,
        "caveat": "",
    }
    row.update(kwargs)
    return row


def _parse_tetrodes(raw: str) -> list[int]:
    return [int(value) for value in re.findall(r"\d+", str(raw))]


def _egf_path_for_channel(stem: Path, channel: int) -> Path:
    if int(channel) == 1:
        return stem.with_suffix(".egf")
    return stem.with_name(f"{stem.name}.egf{int(channel)}")


def _cut_path_for_tetrode(stem: Path, tetrode: int) -> Path | None:
    candidates = [
        stem.with_name(f"{stem.name}_{int(tetrode)}.cut"),
        stem.with_suffix(".cut"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    if np.isfinite(mad) and mad > 0.0:
        return (values - median) / (1.4826 * mad)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if np.isfinite(std) and std > 0.0:
        return (values - mean) / std
    return np.zeros_like(values, dtype=float)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Olafsdottir2016"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--animal", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--min-event-spikes", type=int, default=5)
    parser.add_argument("--min-event-active-cells", type=int, default=3)
    parser.add_argument("--min-duration-s", type=float, default=0.005)
    parser.add_argument("--max-duration-s", type=float, default=0.5)
    parser.add_argument("--ripple-low-hz", type=float, default=150.0)
    parser.add_argument("--ripple-high-hz", type=float, default=250.0)
    parser.add_argument("--ripple-z-threshold", type=float, default=3.2)
    parser.add_argument("--merge-gap-s", type=float, default=0.015)
    parser.add_argument("--envelope-smooth-s", type=float, default=0.008)
    parser.add_argument("--max-lfp-channels", type=int, default=8)
    parser.add_argument("--combine-method", choices=("mean", "max"), default="max")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest_path = args.manifest if args.manifest is not None else args.dataset_root / MANIFEST_NAME
    events, summary = write_detection_outputs(
        load_manifest(manifest_path),
        args.output,
        animal=args.animal,
        date=args.date,
        min_event_spikes=args.min_event_spikes,
        min_event_active_cells=args.min_event_active_cells,
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
        ripple_low_hz=args.ripple_low_hz,
        ripple_high_hz=args.ripple_high_hz,
        ripple_z_threshold=args.ripple_z_threshold,
        merge_gap_s=args.merge_gap_s,
        envelope_smooth_s=args.envelope_smooth_s,
        max_lfp_channels=args.max_lfp_channels,
        combine_method=args.combine_method,
    )
    print(json.dumps({"events": int(len(events)), "sessions": int(len(summary))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
