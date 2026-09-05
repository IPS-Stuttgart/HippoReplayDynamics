"""Pre-decoding event-definition sensitivity with explicit clocks and exclusions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt


@dataclass(frozen=True)
class EventDefinitionConfig:
    maximum_position_gap_s: float = 0.1
    speed_smoothing_s: float = 0.1
    maximum_speed_cm_s: float = 5.0
    fixed_window_s: float = 0.2
    ripple_low_hz: float = 150.0
    ripple_high_hz: float = 250.0
    ripple_filter_order: int = 4
    ripple_envelope_smoothing_s: float = 0.0125
    ripple_peak_z: float = 3.0
    ripple_onset_z: float = 0.0
    ripple_merge_gap_s: float = 0.03
    ripple_minimum_duration_s: float = 0.015
    ripple_maximum_duration_s: float = 0.25
    filter_edge_guard_s: float = 1.0
    minimum_baseline_s: float = 60.0
    maximum_channel_rail_fraction: float = 0.001


def runs(mask):
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 1:
        raise ValueError("run mask must be one dimensional")
    edges = np.diff(np.r_[False, value, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def checked_intervals(intervals):
    value = np.asarray(intervals, dtype=float).reshape(-1, 2)
    if not np.isfinite(value).all() or np.any(value[:, 1] <= value[:, 0]):
        raise ValueError("invalid intervals")
    if len(value) > 1 and np.any(value[1:, 0] < value[:-1, 1]):
        raise ValueError("intervals must be sorted and non-overlapping")
    return value


def interval_contains(starts, ends, intervals):
    intervals = checked_intervals(intervals)
    starts, ends = np.broadcast_arrays(np.asarray(starts, float), np.asarray(ends, float))
    if not len(intervals):
        return np.zeros(starts.shape, bool)
    i = np.searchsorted(intervals[:, 0], starts, side="right") - 1
    valid = (i >= 0) & np.isfinite(starts) & np.isfinite(ends) & (ends >= starts)
    return valid & (ends <= intervals[np.maximum(i, 0), 1] + 1e-9)


def tracking_speed(position, supported_intervals, config):
    """Resample within supported segments; never smooth or interpolate across gaps."""
    position = np.asarray(position, float)
    if position.ndim != 2 or position.shape[1] < 3 or len(position) < 3:
        raise ValueError("missing position")
    if not np.isfinite(position[:, :3]).all() or np.any(np.diff(position[:, 0]) <= 0):
        raise ValueError("position must be finite with increasing clock")
    speed = np.full(len(position), np.nan)
    t = position[:, 0]
    for lo, hi in checked_intervals(supported_intervals):
        a, b = np.searchsorted(t, [lo, hi], side="left")
        b = min(b + int(b < len(t) and t[b] == hi), len(t))
        if b - a < 3:
            continue
        breaks = np.r_[a, a + 1 + np.flatnonzero(np.diff(t[a:b]) > config.maximum_position_gap_s), b]
        for start, stop in pairwise(breaks):
            if stop - start < 3:
                continue
            dt = float(np.median(np.diff(t[start:stop])))
            n = int(np.floor((t[stop - 1] - t[start]) / dt)) + 1
            if n < 3:
                continue
            grid = np.linspace(t[start], t[stop - 1], n)
            xy = np.column_stack([np.interp(grid, t[start:stop], position[start:stop, k]) for k in [1, 2]])
            velocity = np.gradient(xy, grid, axis=0)
            scalar = np.linalg.norm(velocity, axis=1)
            scalar = gaussian_filter1d(scalar, config.speed_smoothing_s / (grid[1] - grid[0]), mode="nearest")
            speed[start:stop] = np.interp(t[start:stop], grid, scalar)
    return speed


def immobile_intervals(times, speed, config):
    """Exact subintervals below threshold for linearly interpolated frame speeds."""
    times, speed = np.asarray(times, float), np.asarray(speed, float)
    if times.shape != speed.shape or times.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError("invalid speed clock")
    intervals = []
    for i in np.flatnonzero(np.isfinite(speed[:-1]) & np.isfinite(speed[1:]) & (np.diff(times) <= config.maximum_position_gap_s)):
        a, b, u, v = times[i], times[i + 1], speed[i], speed[i + 1]
        threshold = config.maximum_speed_cm_s
        if u >= threshold and v >= threshold:
            continue
        if u >= threshold or v >= threshold:
            crossing = a + (b - a) * (threshold - u) / (v - u)
            if u >= threshold:
                a = crossing
            else:
                b = crossing
        if b <= a:
            continue
        if intervals and abs(intervals[-1][1] - a) < 1e-9:
            intervals[-1][1] = b
        else:
            intervals.append([a, b])
    return np.asarray(intervals).reshape(-1, 2)


def validate_lfp_clock(times, sampling_rate):
    times = np.asarray(times, float)
    if times.ndim != 1 or len(times) < 10 or not np.isfinite(times).all():
        raise ValueError("invalid LFP timestamps")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("invalid LFP sampling rate")
    delta = np.diff(times)
    if not np.allclose(delta, 1 / sampling_rate, rtol=1e-4, atol=1e-9):
        raise ValueError("LFP clock is irregular or has a gap; no implicit repair")


def ripple_envelope(data, sampling_rate, config):
    data = np.asarray(data)
    if data.ndim != 1 or not np.isfinite(data).all() or np.ptp(data.astype(float)) == 0:
        raise ValueError("nonfinite or constant LFP channel")
    if sampling_rate <= 2 * config.ripple_high_hz:
        raise ValueError("LFP Nyquist frequency too low")
    sos = butter(config.ripple_filter_order, [config.ripple_low_hz, config.ripple_high_hz], btype="bandpass", fs=sampling_rate, output="sos")
    filtered = sosfiltfilt(sos, data.astype(float))
    return gaussian_filter1d(np.abs(hilbert(filtered)), config.ripple_envelope_smoothing_s * sampling_rate, mode="nearest")


def standardize_envelope(envelope, times, sampling_rate, immobile, config):
    validate_lfp_clock(times, sampling_rate)
    envelope = np.asarray(envelope, float)
    if envelope.shape != times.shape or not np.isfinite(envelope).all():
        raise ValueError("invalid ripple envelope")
    edge_valid = (times >= times[0] + config.filter_edge_guard_s) & (times < times[-1] - config.filter_edge_guard_s)
    baseline = edge_valid & interval_contains(times, times, immobile)
    if baseline.sum() / sampling_rate < config.minimum_baseline_s:
        raise ValueError("insufficient tracking-supported immobile LFP baseline")
    mean, sd = float(np.mean(envelope[baseline])), float(np.std(envelope[baseline]))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("degenerate ripple baseline")
    z = (envelope - mean) / sd
    z[~edge_valid] = np.nan
    return z, baseline, {"envelope_baseline_mean": mean, "envelope_baseline_sd": sd, "baseline_duration_s": float(baseline.sum() / sampling_rate)}


def detect_ripple_episodes(times, z, sampling_rate, config):
    """Keep every amplitude-qualified episode, with a separate duration flag."""
    validate_lfp_clock(times, sampling_rate)
    z = np.asarray(z, float)
    if z.shape != times.shape:
        raise ValueError("envelope/clock mismatch")
    merged = []
    for a, b in runs(np.isfinite(z) & (z > config.ripple_onset_z)):
        if merged and (a - merged[-1][1]) / sampling_rate <= config.ripple_merge_gap_s + 1e-12 and np.isfinite(z[merged[-1][1]:a]).all():
            merged[-1][1] = b
        else:
            merged.append([a, b])
    rows = []
    for episode_id, (a, b) in enumerate(merged):
        peak = a + int(np.argmax(z[a:b]))
        if z[peak] < config.ripple_peak_z:
            continue
        duration = (b - a) / sampling_rate
        rows.append({"source_event_id": episode_id, "start_s": float(times[a]),
                     "end_s": float(times[b - 1] + 1 / sampling_rate), "peak_s": float(times[peak]),
                     "ripple_peak_z": float(z[peak]), "core_duration_s": float(duration),
                     "detector_duration_pass": bool(config.ripple_minimum_duration_s <= duration <= config.ripple_maximum_duration_s),
                     "lfp_start_sample": a, "lfp_end_sample_exclusive": b})
    return pd.DataFrame(rows, columns=["source_event_id", "start_s", "end_s", "peak_s", "ripple_peak_z", "core_duration_s", "detector_duration_pass", "lfp_start_sample", "lfp_end_sample_exclusive"])


def source_events(mua, ripple, dataset, animal, session):
    mua, ripple = mua.copy(), ripple.copy()
    mua = mua.rename(columns={"event_index": "source_event_id"})
    mua["detector"] = "source_high_mua"
    mua["detector_duration_pass"] = True
    ripple["detector"] = "native_ripple_table" if dataset == "pfeiffer_foster" else "lfp_ripple_detected"
    if "detector_duration_pass" not in ripple:
        ripple["detector_duration_pass"] = True
    rows = pd.concat([mua, ripple], ignore_index=True)
    for key, value in [("dataset", dataset), ("animal", animal), ("session", session)]:
        rows[key] = value
    required = ["start_s", "end_s", "peak_s", "source_event_id"]
    if not np.isfinite(rows[required].to_numpy(float)).all():
        raise ValueError("nonfinite event boundaries or IDs")
    if (rows.end_s <= rows.start_s).any() or (rows.peak_s < rows.start_s).any() or (rows.peak_s >= rows.end_s).any():
        raise ValueError("invalid event boundaries/peak")
    if (rows.source_event_id != np.floor(rows.source_event_id)).any() or rows.duplicated(["detector", "source_event_id"]).any():
        raise ValueError("invalid or duplicate source event identity")
    rows["source_event_id"] = rows.source_event_id.astype(int)
    prefix = f"{dataset}:{animal}:{session}"
    rows["source_uid"] = [f"{prefix}:{d}:{i}" for d, i in zip(rows.detector, rows.source_event_id, strict=True)]
    return rows.sort_values(["detector", "source_event_id"]).reset_index(drop=True)


def make_windows(events, position, supported, immobile, config, lfp_times=None):
    records = []
    for variant in ["detected_core", "peak_centered_200ms"]:
        frame = events.copy()
        frame["window_variant"] = variant
        frame["core_start_s"], frame["core_end_s"] = frame.start_s, frame.end_s
        if variant == "peak_centered_200ms":
            frame["start_s"] = frame.peak_s - config.fixed_window_s / 2
            frame["end_s"] = frame.peak_s + config.fixed_window_s / 2
        frame["window_uid"] = frame.source_uid + ":" + variant
        frame["window_duration_s"] = frame.end_s - frame.start_s
        frame["position_clock_valid"] = (frame.start_s >= position[0, 0]) & (frame.end_s <= position[-1, 0])
        frame["tracking_supported"] = interval_contains(frame.start_s, frame.end_s, supported)
        frame["whole_window_immobile"] = interval_contains(frame.start_s, frame.end_s, immobile)
        frame["peak_immobile"] = interval_contains(frame.peak_s, frame.peak_s, immobile)
        frame["lfp_window_supported"] = True if lfp_times is None else ((frame.start_s >= lfp_times[0] + config.filter_edge_guard_s) & (frame.end_s <= lfp_times[-1] - config.filter_edge_guard_s))
        flags = ["detector_duration_pass", "position_clock_valid", "tracking_supported", "whole_window_immobile", "lfp_window_supported"]
        frame["eligible"] = frame[flags].all(axis=1)
        frame["exclusion_reason"] = [";".join(k for k in flags if not getattr(row, k)) for row in frame.itertuples(index=False)]
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def overlap_pairs(windows):
    """Many-to-many interval overlaps; counts never treat an overlap as a new event."""
    rows = []
    keys = ["dataset", "animal", "session", "window_variant"]
    for identity, group in windows.groupby(keys, sort=True):
        eligible = group[group.eligible]
        mua = eligible[eligible.detector.eq("source_high_mua")]
        ripple = eligible[~eligible.detector.eq("source_high_mua")]
        for a in mua.itertuples(index=False):
            matches = ripple[(ripple.start_s < a.end_s) & (ripple.end_s > a.start_s)]
            for b in matches.itertuples(index=False):
                rows.append({**dict(zip(keys, identity, strict=True)), "mua_window_uid": a.window_uid, "ripple_window_uid": b.window_uid,
                    "overlap_s": min(a.end_s, b.end_s) - max(a.start_s, b.start_s), "peak_offset_s": b.peak_s - a.peak_s})
    return pd.DataFrame(rows, columns=keys + ["mua_window_uid", "ripple_window_uid", "overlap_s", "peak_offset_s"])


def annotate_overlap(windows, pairs):
    windows = windows.copy()
    counts = pd.concat([pairs.mua_window_uid, pairs.ripple_window_uid]).value_counts()
    windows["other_detector_overlap_count"] = windows.window_uid.map(counts).fillna(0).astype(int)
    windows["overlap_class"] = np.where(~windows.eligible, "ineligible", np.where(windows.other_detector_overlap_count > 0, "both_detectors", np.where(windows.detector.eq("source_high_mua"), "mua_only", "ripple_only")))
    return windows


def window_spike_support(windows, indexed, cell_ids, qc_mask):
    cell_ids, qc_mask = np.asarray(cell_ids), np.asarray(qc_mask, bool)
    counts = np.empty((len(windows), len(cell_ids)), np.int64)
    for j, cell in enumerate(cell_ids):
        times = indexed[int(cell)]
        counts[:, j] = np.searchsorted(times, windows.end_s, side="left") - np.searchsorted(times, windows.start_s, side="left")
    return pd.DataFrame({"n_spikes_all_sorted": counts.sum(axis=1), "n_spikes_qc_units": counts[:, qc_mask].sum(axis=1),
        "n_active_qc_units": (counts[:, qc_mask] > 0).sum(axis=1)})


def ripple_window_metrics(windows, times, z):
    rows = []
    for row in windows.itertuples(index=False):
        a, b = np.searchsorted(times, [row.start_s, row.end_s], side="left")
        local = z[a:b]
        measured = local[np.isfinite(local)]
        rows.append({"mean_ripple_envelope_z": float(np.mean(measured)) if len(measured) else np.nan,
            "maximum_ripple_envelope_z": float(np.max(measured)) if len(measured) else np.nan,
            "fraction_ripple_samples_above_z3": float(np.mean(measured >= 3)) if len(measured) else np.nan,
            "ripple_samples_measured": len(measured)})
    return pd.DataFrame(rows)
