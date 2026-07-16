#!/usr/bin/env python3
"""Detect hc-11 NREM ripples from raw multiplexed EEG files.

This adapter reconstructs the published ``bz_FindRipples`` detector closely:
130-200 Hz third-order Butterworth filtering, per-channel squared power summed
across tagged CA1 ripple channels, 11-sample smoothing, 2/5 SD thresholds,
20 ms event merging, and 20-300 ms duration limits. Channels are filtered
separately before power is combined, so a 180-degree phase reversal cannot
cancel a ripple as it would under raw-channel averaging.

When a published ripple table is available, the script reports one-to-one
precision/recall against it. Generated tables are written to an explicit output
path and are never installed into the processed dataset silently.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _provenance import build_script_provenance  # noqa: E402
import score_hc11_pre_post_learning_evidence as learning  # noqa: E402
import score_hc11_webshare_native_ripple_evidence as hc11  # noqa: E402


EVENT_OUTPUT = "hc11_lfp_detected_ripples.csv"
QC_OUTPUT = "hc11_lfp_ripple_detection_qc.csv"
MANIFEST_OUTPUT = "hc11_lfp_ripple_detection_manifest.json"


def select_ca1_shank_channels(
    anatomical_groups: list[np.ndarray],
    channel_regions: np.ndarray,
    bad_channels: np.ndarray,
    unit_shanks: np.ndarray,
    unit_channels: np.ndarray,
    unit_regions: np.ndarray,
) -> np.ndarray:
    """Choose one pyramidal-layer proxy channel per CA1 anatomical group."""

    regions = np.asarray(channel_regions, dtype=object).ravel()
    bad = set(int(value) for value in np.asarray(bad_channels, dtype=int).ravel())
    shanks = np.asarray(unit_shanks, dtype=int).ravel()
    max_channels = np.asarray(unit_channels, dtype=int).ravel()
    cell_regions = np.asarray(unit_regions, dtype=object).ravel()
    selected: list[int] = []
    for group_index, raw_group in enumerate(anatomical_groups, start=1):
        group = np.asarray(raw_group, dtype=int).ravel()
        group = group[(group >= 0) & (group < len(regions))]
        ca1_group = np.asarray(
            [channel for channel in group if "CA1" in str(regions[channel]).upper()],
            dtype=int,
        )
        good = np.asarray([channel for channel in ca1_group if int(channel) not in bad], dtype=int)
        if good.size == 0:
            continue
        midpoint = float(np.mean(good))
        counts: dict[int, int] = {}
        for shank, channel, region in zip(shanks, max_channels, cell_regions, strict=True):
            if (
                int(shank) == group_index
                and int(channel) in set(int(value) for value in good)
                and "CA1" in str(region).upper()
            ):
                counts[int(channel)] = counts.get(int(channel), 0) + 1
        if counts:
            choice = min(
                counts,
                key=lambda channel: (-counts[channel], abs(channel - midpoint), channel),
            )
        else:
            choice = min((int(value) for value in good), key=lambda channel: (abs(channel - midpoint), channel))
        selected.append(choice)
    channels = np.unique(np.asarray(selected, dtype=int))
    if channels.size == 0:
        raise ValueError("could not derive any CA1 ripple channels from anatomical groups")
    return channels


def _fallback_ripple_channels(session_dir: Path, info: object) -> np.ndarray:
    base = session_dir.name
    spikes = loadmat(
        session_dir / f"{base}.spikes.cellinfo.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["spikes"]
    groups = [np.asarray(group.Channels, dtype=int).ravel() for group in np.asarray(info.AnatGrps).ravel()]
    return select_ca1_shank_channels(
        groups,
        np.asarray(info.region, dtype=object).ravel(),
        np.asarray(getattr(info, "badchannels", []), dtype=int).ravel(),
        np.asarray(spikes.shankID, dtype=int).ravel(),
        np.asarray(spikes.maxWaveformCh, dtype=int).ravel(),
        np.asarray(spikes.region, dtype=object).ravel(),
    )


def load_eeg_metadata(
    session_dir: Path,
    *,
    channel_source: str = "auto",
) -> tuple[int, float, np.ndarray, str]:
    base = session_dir.name
    info = loadmat(
        session_dir / f"{base}.sessionInfo.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["sessionInfo"]
    tagged = (
        np.asarray(info.channelTags.ripchans, dtype=int).ravel()
        if hasattr(info, "channelTags") and hasattr(info.channelTags, "ripchans")
        else np.array([], dtype=int)
    )
    if channel_source == "tagged":
        if tagged.size == 0:
            raise ValueError(f"{base}: sessionInfo.channelTags.ripchans is unavailable")
        channels = tagged
        resolved_source = "sessionInfo.channelTags.ripchans"
    elif channel_source == "ca1_shank_fallback":
        channels = _fallback_ripple_channels(session_dir, info)
        resolved_source = "ca1_shank_unit_mode_fallback"
    elif channel_source == "auto":
        if tagged.size:
            channels = tagged
            resolved_source = "sessionInfo.channelTags.ripchans"
        else:
            channels = _fallback_ripple_channels(session_dir, info)
            resolved_source = "ca1_shank_unit_mode_fallback"
    else:
        raise ValueError("channel_source must be auto, tagged, or ca1_shank_fallback")
    channels = np.unique(channels[channels >= 0])
    if channels.size == 0:
        raise ValueError(f"{base}: resolved ripple-channel list is empty")
    n_channels = int(info.nChannels)
    if np.any(channels >= n_channels):
        raise ValueError(f"{base}: ripple channel lies outside 0..{n_channels - 1}")
    return n_channels, float(info.lfpSampleRate), channels, resolved_source


def ripple_power_nss(
    signals: np.ndarray,
    sample_rate_hz: float,
    *,
    passband_hz: tuple[float, float] = (130.0, 200.0),
    smoothing_samples: int = 11,
) -> np.ndarray:
    """Return smoothed summed ripple-band power without raw-channel averaging."""

    values = np.asarray(signals, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 16:
        raise ValueError("signals must be time by channel with at least 16 samples")
    low, high = map(float, passband_hz)
    if not 0.0 < low < high < 0.5 * float(sample_rate_hz):
        raise ValueError("passband must lie inside the Nyquist interval")
    sos = butter(3, (low, high), btype="bandpass", fs=float(sample_rate_hz), output="sos")
    filtered = sosfiltfilt(sos, values, axis=0)
    squared_sum = np.sum(np.square(filtered), axis=1)
    return uniform_filter1d(
        squared_sum,
        size=int(smoothing_samples),
        mode="nearest",
    )


def build_power_memmap(
    eeg_path: Path,
    output_path: Path,
    *,
    n_channels: int,
    ripple_channels: np.ndarray,
    sample_rate_hz: float,
    passband_hz: tuple[float, float],
    chunk_seconds: float,
    overlap_seconds: float = 2.0,
) -> np.memmap:
    bytes_per_frame = np.dtype("<i2").itemsize * int(n_channels)
    file_size = eeg_path.stat().st_size
    if file_size % bytes_per_frame:
        raise ValueError(
            f"{eeg_path}: size {file_size} is not divisible by {bytes_per_frame} bytes/frame"
        )
    n_samples = file_size // bytes_per_frame
    raw = np.memmap(eeg_path, dtype="<i2", mode="r", shape=(n_samples, int(n_channels)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    power = np.memmap(output_path, dtype="<f4", mode="w+", shape=(n_samples,))
    chunk = max(int(float(chunk_seconds) * sample_rate_hz), 1)
    overlap = max(int(float(overlap_seconds) * sample_rate_hz), 32)
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        read_start = max(0, start - overlap)
        read_stop = min(n_samples, stop + overlap)
        values = np.asarray(raw[read_start:read_stop, ripple_channels], dtype=np.float64)
        local = ripple_power_nss(
            values,
            sample_rate_hz,
            passband_hz=passband_hz,
        )
        power[start:stop] = local[start - read_start : stop - read_start].astype(np.float32)
    power.flush()
    return power


def interval_sample_bounds(
    intervals: np.ndarray,
    sample_rate_hz: float,
    n_samples: int,
) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for start_s, end_s in np.asarray(intervals, dtype=float).reshape(-1, 2):
        start = max(int(np.ceil(float(start_s) * sample_rate_hz)), 0)
        stop = min(int(np.floor(float(end_s) * sample_rate_hz)), int(n_samples))
        if stop > start:
            bounds.append((start, stop))
    return bounds


def restricted_mean_sd(
    values: np.ndarray,
    bounds: list[tuple[int, int]],
) -> tuple[float, float, int]:
    total = 0
    value_sum = 0.0
    square_sum = 0.0
    for start, stop in bounds:
        chunk = np.asarray(values[start:stop], dtype=np.float64)
        finite = chunk[np.isfinite(chunk)]
        total += len(finite)
        value_sum += float(finite.sum())
        square_sum += float(np.dot(finite, finite))
    if total < 2:
        raise ValueError("normalization intervals contain fewer than two finite samples")
    mean = value_sum / total
    variance = max((square_sum - total * mean * mean) / (total - 1), 0.0)
    sd = float(np.sqrt(variance))
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError("restricted ripple-power standard deviation is not positive")
    return mean, sd, total


def _threshold_runs(mask: np.ndarray, offset: int) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1) + int(offset)
    stops = np.flatnonzero(changes == -1) + int(offset)
    return list(zip(starts, stops, strict=True))


def detect_threshold_events(
    normalized_power: np.ndarray,
    bounds: list[tuple[int, int]],
    sample_rate_hz: float,
    *,
    low_threshold: float = 2.0,
    high_threshold: float = 5.0,
    min_inter_event_ms: float = 20.0,
    min_duration_ms: float = 20.0,
    max_duration_ms: float = 300.0,
) -> pd.DataFrame:
    """Apply the bz_FindRipples threshold, merge, peak, and duration stages."""

    values = np.asarray(normalized_power, dtype=float)
    merge_samples = int(np.rint(float(min_inter_event_ms) * sample_rate_hz / 1000.0))
    min_samples = int(np.ceil(float(min_duration_ms) * sample_rate_hz / 1000.0))
    max_samples = int(np.floor(float(max_duration_ms) * sample_rate_hz / 1000.0))
    rows: list[dict[str, object]] = []
    event_id = 0
    for interval_index, (bound_start, bound_stop) in enumerate(bounds):
        runs = _threshold_runs(values[bound_start:bound_stop] > float(low_threshold), bound_start)
        merged: list[list[int]] = []
        for start, stop in runs:
            if merged and start - merged[-1][1] < merge_samples:
                merged[-1][1] = stop
            else:
                merged.append([start, stop])
        for start, stop in merged:
            duration_samples = stop - start
            if duration_samples < min_samples or duration_samples > max_samples:
                continue
            local = values[start:stop]
            if local.size == 0:
                continue
            peak_offset = int(np.argmax(local))
            peak_value = float(local[peak_offset])
            if peak_value <= float(high_threshold):
                continue
            rows.append(
                {
                    "event_id": event_id,
                    "start_time_s": start / float(sample_rate_hz),
                    "end_time_s": stop / float(sample_rate_hz),
                    "peak_time_s": (start + peak_offset) / float(sample_rate_hz),
                    "duration_ms": 1000.0 * duration_samples / float(sample_rate_hz),
                    "peak_normed_power": peak_value,
                    "nrem_interval_index": interval_index,
                }
            )
            event_id += 1
    return pd.DataFrame(rows)


def compare_peak_times(
    detected_peaks: np.ndarray,
    native_peaks: np.ndarray,
    tolerance_s: float,
) -> dict[str, float | int]:
    detected = np.asarray(detected_peaks, dtype=float).ravel()
    native = np.asarray(native_peaks, dtype=float).ravel()
    if detected.size == 0 or native.size == 0:
        return {
            "detected_events": int(len(detected)),
            "native_events": int(len(native)),
            "matched_events": 0,
            "precision": 0.0,
            "recall": 0.0,
            "median_absolute_peak_error_ms": np.nan,
        }
    cost = np.abs(detected[:, None] - native[None, :])
    detected_index, native_index = linear_sum_assignment(cost)
    errors = cost[detected_index, native_index]
    keep = errors <= float(tolerance_s)
    matched = int(np.sum(keep))
    return {
        "detected_events": int(len(detected)),
        "native_events": int(len(native)),
        "matched_events": matched,
        "precision": matched / len(detected),
        "recall": matched / len(native),
        "median_absolute_peak_error_ms": (
            float(1000.0 * np.median(errors[keep])) if matched else np.nan
        ),
    }


def compare_event_intervals(
    detected_intervals: np.ndarray,
    native_intervals: np.ndarray,
    *,
    min_iou: float = 0.0,
) -> dict[str, float | int]:
    """Match detected and native events one-to-one by interval IoU."""

    detected = np.asarray(detected_intervals, dtype=float).reshape(-1, 2)
    native = np.asarray(native_intervals, dtype=float).reshape(-1, 2)
    if not 0.0 <= float(min_iou) <= 1.0:
        raise ValueError("min_iou must lie in [0, 1]")
    if detected.size == 0 or native.size == 0:
        return {
            "matched_events": 0,
            "precision": 0.0,
            "recall": 0.0,
            "median_iou": np.nan,
        }
    intersection = np.maximum(
        0.0,
        np.minimum(detected[:, None, 1], native[None, :, 1])
        - np.maximum(detected[:, None, 0], native[None, :, 0]),
    )
    union = (
        np.maximum(detected[:, None, 1], native[None, :, 1])
        - np.minimum(detected[:, None, 0], native[None, :, 0])
    )
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    detected_index, native_index = linear_sum_assignment(1.0 - iou)
    matched_iou = iou[detected_index, native_index]
    if min_iou == 0.0:
        keep = matched_iou > 0.0
    else:
        keep = matched_iou >= float(min_iou)
    matched = int(np.sum(keep))
    return {
        "matched_events": matched,
        "precision": matched / len(detected),
        "recall": matched / len(native),
        "median_iou": float(np.median(matched_iou[keep])) if matched else np.nan,
    }


def load_native_restrict_intervals(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        key = "ripplesNREM/detectorparms/restrict"
        if key not in handle:
            return np.empty((0, 2), dtype=float)
        values = np.asarray(handle[key], dtype=float)
    if values.ndim != 2:
        return np.empty((0, 2), dtype=float)
    if values.shape[0] == 2:
        values = values.T
    elif values.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    keep = np.isfinite(values).all(axis=1) & (values[:, 1] > values[:, 0])
    return values[keep]


def write_event_mat(
    path: Path,
    events: pd.DataFrame,
    *,
    sample_rate_hz: float,
    ripple_channels: np.ndarray,
    normalization_sd: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("ripplesNREM")
        group.create_dataset(
            "times",
            data=np.vstack(
                [
                    events["start_time_s"].to_numpy(dtype=float),
                    events["end_time_s"].to_numpy(dtype=float),
                ]
            ),
        )
        group.create_dataset("peaks", data=events["peak_time_s"].to_numpy(dtype=float)[None, :])
        group.create_dataset(
            "peakNormedPower",
            data=events["peak_normed_power"].to_numpy(dtype=float)[None, :],
        )
        group.create_dataset("stdev", data=np.array([[float(normalization_sd)]]))
        group.create_dataset(
            "detectorName",
            data=np.asarray([ord(value) for value in "python_bz_FindRipples_multichannel"], dtype=np.uint16)[:, None],
        )
        params = group.create_group("detectorparms")
        params.create_dataset("frequency", data=np.array([[float(sample_rate_hz)]]))
        params.create_dataset("thresholds", data=np.array([[args.low_threshold], [args.high_threshold]]))
        params.create_dataset("durations", data=np.array([[args.min_inter_event_ms], [args.max_duration_ms]]))
        params.create_dataset("minDuration", data=np.array([[args.min_duration_ms]]))
        params.create_dataset("passband", data=np.array(args.passband_hz, dtype=float)[:, None])
        params.create_dataset("channels", data=np.asarray(ripple_channels, dtype=int)[:, None])


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_dir = Path(args.session_dir).resolve()
    eeg_path = Path(args.eeg_path).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_channels, sample_rate_hz, ripple_channels, ripple_channel_source = load_eeg_metadata(
        session_dir,
        channel_source=args.ripple_channel_source,
    )
    power_path = output_dir / f"{session_dir.name}.ripple_power.float32"
    frame_bytes = np.dtype("<i2").itemsize * n_channels
    n_samples = eeg_path.stat().st_size // frame_bytes
    expected_power_bytes = n_samples * np.dtype("<f4").itemsize
    if (
        args.reuse_power_file
        and power_path.exists()
        and power_path.stat().st_size == expected_power_bytes
    ):
        power = np.memmap(power_path, dtype="<f4", mode="r", shape=(n_samples,))
    else:
        power = build_power_memmap(
            eeg_path,
            power_path,
            n_channels=n_channels,
            ripple_channels=ripple_channels,
            sample_rate_hz=sample_rate_hz,
            passband_hz=tuple(args.passband_hz),
            chunk_seconds=args.chunk_seconds,
        )
    native_path = session_dir / f"{session_dir.name}.ripplesNREM.event.mat"
    native_restrict = load_native_restrict_intervals(native_path) if native_path.exists() else np.empty((0, 2))
    if args.interval_source == "native_restrict" and native_restrict.size == 0:
        raise ValueError("native_restrict requested but the published detector intervals are unavailable")
    use_native_restrict = args.interval_source == "native_restrict" or (
        args.interval_source == "auto" and native_restrict.size > 0
    )
    detection_intervals = native_restrict if use_native_restrict else learning.nrem_intervals(session_dir)
    interval_source = "native_restrict" if use_native_restrict else "nrem_state"
    bounds = interval_sample_bounds(detection_intervals, sample_rate_hz, len(power))
    mean, sd, normalization_samples = restricted_mean_sd(power, bounds)
    normalized_path = output_dir / f"{session_dir.name}.ripple_power_z.float32"
    normalized = np.memmap(normalized_path, dtype="<f4", mode="w+", shape=power.shape)
    chunk = max(int(args.chunk_seconds * sample_rate_hz), 1)
    for start in range(0, len(power), chunk):
        stop = min(start + chunk, len(power))
        normalized[start:stop] = ((power[start:stop] - mean) / sd).astype(np.float32)
    normalized.flush()
    events = detect_threshold_events(
        normalized,
        bounds,
        sample_rate_hz,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        min_inter_event_ms=args.min_inter_event_ms,
        min_duration_ms=args.min_duration_ms,
        max_duration_ms=args.max_duration_ms,
    )
    events.insert(0, "session", session_dir.name)
    events.insert(0, "animal", session_dir.parent.name)
    events.to_csv(output_dir / EVENT_OUTPUT, index=False)

    comparison: dict[str, float | int] = {
        "detected_events": int(len(events)),
        "native_events": 0,
        "matched_events": 0,
        "precision": np.nan,
        "recall": np.nan,
        "median_absolute_peak_error_ms": np.nan,
        "overlap_matched_events": 0,
        "overlap_precision": np.nan,
        "overlap_recall": np.nan,
        "median_overlap_iou": np.nan,
        "iou25_matched_events": 0,
        "iou25_precision": np.nan,
        "iou25_recall": np.nan,
        "median_iou25": np.nan,
    }
    if native_path.exists():
        with h5py.File(native_path, "r") as handle:
            native_peaks = np.asarray(handle["ripplesNREM/peaks"], dtype=float).ravel()
            native_times = np.asarray(handle["ripplesNREM/times"], dtype=float)
        if native_times.ndim != 2:
            raise ValueError(f"{native_path}: ripplesNREM/times must be two-dimensional")
        if native_times.shape[0] == 2:
            native_times = native_times.T
        elif native_times.shape[1] != 2:
            raise ValueError(f"{native_path}: ripplesNREM/times must have two columns")
        if len(native_times) != len(native_peaks):
            raise ValueError(f"{native_path}: ripple times and peaks have different lengths")
        recording_duration_s = len(power) / float(sample_rate_hz)
        keep = (
            np.isfinite(native_peaks)
            & (native_peaks >= 0.0)
            & (native_peaks < recording_duration_s)
            & np.isfinite(native_times).all(axis=1)
        )
        native_peaks = native_peaks[keep]
        native_times = native_times[keep]
        if detection_intervals.size:
            keep = hc11.times_in_intervals(native_peaks, detection_intervals)
            native_peaks = native_peaks[keep]
            native_times = native_times[keep]
        comparison = compare_peak_times(
            events["peak_time_s"].to_numpy(dtype=float),
            native_peaks,
            args.validation_tolerance_ms / 1000.0,
        )
        detected_times = events[["start_time_s", "end_time_s"]].to_numpy(dtype=float)
        overlap = compare_event_intervals(detected_times, native_times)
        iou25 = compare_event_intervals(detected_times, native_times, min_iou=0.25)
        comparison.update(
            {
                "overlap_matched_events": overlap["matched_events"],
                "overlap_precision": overlap["precision"],
                "overlap_recall": overlap["recall"],
                "median_overlap_iou": overlap["median_iou"],
                "iou25_matched_events": iou25["matched_events"],
                "iou25_precision": iou25["precision"],
                "iou25_recall": iou25["recall"],
                "median_iou25": iou25["median_iou"],
            }
        )
    qc = pd.DataFrame(
        [
            {
                "animal": session_dir.parent.name,
                "session": session_dir.name,
                "eeg_path": str(eeg_path),
                "n_channels": n_channels,
                "ripple_channels": ",".join(str(value) for value in ripple_channels),
                "ripple_channel_source": ripple_channel_source,
                "sample_rate_hz": sample_rate_hz,
                "detection_interval_source": interval_source,
                "nrem_intervals": len(bounds),
                "normalization_samples": normalization_samples,
                "normalization_mean": mean,
                "normalization_sd": sd,
                **comparison,
                "native_validation_available": native_path.exists(),
            }
        ]
    )
    qc.to_csv(output_dir / QC_OUTPUT, index=False)
    if args.output_mat:
        write_event_mat(
            Path(args.output_mat),
            events,
            sample_rate_hz=sample_rate_hz,
            ripple_channels=ripple_channels,
            normalization_sd=sd,
            args=args,
        )
    provenance = build_script_provenance(
        input_paths={
            "session_info": session_dir / f"{session_dir.name}.sessionInfo.mat",
            "native_ripple_table": native_path if native_path.exists() else None,
        },
        cwd=ROOT,
        argv=sys.argv,
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session_dir),
        "eeg_path": str(eeg_path),
        "eeg_size_bytes": eeg_path.stat().st_size,
        "output_mat": str(Path(args.output_mat).resolve()) if args.output_mat else None,
        "channel_combination": "filter_each_then_sum_squared_power",
        "ripple_channel_source": ripple_channel_source,
        "parameters": {
            "passband_hz": list(args.passband_hz),
            "thresholds_sd": [args.low_threshold, args.high_threshold],
            "min_inter_event_ms": args.min_inter_event_ms,
            "min_duration_ms": args.min_duration_ms,
            "max_duration_ms": args.max_duration_ms,
        },
        **provenance,
    }
    (output_dir / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not args.keep_power_files:
        del normalized
        del power
        normalized_path.unlink(missing_ok=True)
        power_path.unlink(missing_ok=True)
    return events, qc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--eeg-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-mat")
    parser.add_argument("--passband-hz", type=float, nargs=2, default=(130.0, 200.0))
    parser.add_argument(
        "--ripple-channel-source",
        choices=("auto", "tagged", "ca1_shank_fallback"),
        default="auto",
        help="Use published ripple-channel tags when available or a documented CA1-shank fallback.",
    )
    parser.add_argument("--low-threshold", type=float, default=2.0)
    parser.add_argument("--high-threshold", type=float, default=5.0)
    parser.add_argument("--min-inter-event-ms", type=float, default=20.0)
    parser.add_argument("--min-duration-ms", type=float, default=20.0)
    parser.add_argument("--max-duration-ms", type=float, default=300.0)
    parser.add_argument("--validation-tolerance-ms", type=float, default=25.0)
    parser.add_argument("--chunk-seconds", type=float, default=600.0)
    parser.add_argument(
        "--interval-source",
        choices=("auto", "nrem", "native_restrict"),
        default="auto",
        help="Auto uses published detector intervals for validation and NREM state intervals otherwise.",
    )
    parser.add_argument("--reuse-power-file", action="store_true")
    parser.add_argument("--keep-power-files", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    events, qc = run(args)
    print(f"Detected {len(events)} NREM ripples")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
