#!/usr/bin/env python3
"""Test wall-distance dependence of awake ripple-associated decoded speed.

This analysis intentionally reports two metrics:

* physical posterior-mean speed in cm/s;
* Poisson/Hellinger population-code speed in sqrt(Hz)/s.

The second metric, local code-gradient controls, held-out RUN decoding error, and
a constant-physical-speed decoder simulation separate replay dynamics from the
known wall-dependent spatial resolution of the place-cell code.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import spearmanr

from hipporeplayimm.data import ReplaySession, RippleEvent
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, EncodingModel, build_emissions, fit_place_field_encoding
from hipporeplayimm.tanni2022 import (
    ARCHIVE_MD5,
    ARCHIVE_URL,
    DATASET_DOI,
    PAPER_DOI,
    TanniPosition,
    aggregate_ripple_envelope_z,
    detect_ripple_candidates,
    local_poisson_code_gradient,
    nearest_wall_distance,
    posterior_from_log_likelihood,
    posterior_path_segments,
    read_tanni_position,
    read_tanni_session_metadata,
    read_tanni_sorted_spikes,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import build_script_provenance  # noqa: E402


REQUIRED_OUTPUTS = (
    "tanni2022_session_manifest.csv",
    "tanni2022_unit_qc.csv",
    "tanni2022_decoder_qc_samples.csv",
    "tanni2022_decoder_qc_summary.csv",
    "tanni2022_ripple_candidates.csv",
    "tanni2022_replay_speed_events.csv",
    "tanni2022_replay_speed_segments.csv",
    "tanni2022_wall_distance_quartiles.csv",
    "tanni2022_wall_distance_associations.csv",
    "tanni2022_synthetic_constant_speed_null.csv",
    "tanni2022_wall_distance_gate_summary.csv",
    "tanni2022_wall_distance_replay_figure.png",
    "tanni2022_wall_distance_replay_summary.md",
    "tanni2022_wall_distance_manifest.json",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nwb-glob", default="**/experiment_1.nwb")
    parser.add_argument("--bin-size-cm", type=float, default=8.0)
    parser.add_argument("--rate-smoothing-sigma-bins", type=float, default=1.5)
    parser.add_argument("--running-speed-cm-s", type=float, default=10.0)
    parser.add_argument("--min-running-spikes", type=int, default=30)
    parser.add_argument("--max-mean-rate-hz", type=float, default=4.0)
    parser.add_argument("--min-peak-rate-hz", type=float, default=2.0)
    parser.add_argument("--min-split-half-stability", type=float, default=0.25)
    parser.add_argument("--ripple-threshold-z", type=float, default=3.0)
    parser.add_argument("--ripple-peak-threshold-z", type=float, default=10.0)
    parser.add_argument("--ripple-min-duration-s", type=float, default=0.015)
    parser.add_argument("--ripple-max-duration-s", type=float, default=0.250)
    parser.add_argument("--ripple-merge-gap-s", type=float, default=0.030)
    parser.add_argument("--immobility-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--event-half-window-s", type=float, default=0.100)
    parser.add_argument("--min-event-spikes", type=int, default=5)
    parser.add_argument("--min-event-active-cells", type=int, default=3)
    parser.add_argument("--decode-bin-s", type=float, default=0.020)
    parser.add_argument("--decoder-window-s", type=float, default=0.250)
    parser.add_argument("--decoder-folds", type=int, default=5)
    parser.add_argument("--decoder-samples-per-fold", type=int, default=250)
    parser.add_argument("--synthetic-events-per-session", type=int, default=200)
    parser.add_argument("--synthetic-speed-cm-s", type=float, default=1000.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20220714)
    parser.add_argument("--max-lfp-channels", type=int)
    parser.add_argument("--reuse-ripple-envelope", action="store_true")
    return parser.parse_args(argv)


def make_replay_session(path: Path, position: TanniPosition) -> ReplaySession:
    metadata = read_tanni_session_metadata(path)
    spike_data = read_tanni_sorted_spikes(path)
    start = float(max(position.times_s[0], metadata.lfp_start_time_s))
    end = float(min(position.times_s[-1], metadata.lfp_end_time_s))
    spikes = spike_data.spikes
    spikes = spikes[(spikes[:, 0] >= start) & (spikes[:, 0] <= end)] if spikes.size else spikes
    cell_ids = np.unique(spikes[:, 1].astype(int)) if spikes.size else np.empty(0, dtype=int)
    return ReplaySession(
        rat=metadata.animal,
        name=metadata.session,
        path=path.parent,
        position=np.column_stack((position.times_s, position.xy_cm)),
        spikes=spikes,
        tetrode_cell_ids=np.column_stack((cell_ids // 1000, cell_ids % 1000)) if cell_ids.size else np.empty((0, 2), dtype=int),
        excitatory_neurons=cell_ids,
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[start, end]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={"arena_size_cm": metadata.arena_size_cm.tolist(), "source_dataset": "tanni2022"},
    )


def fit_decoder_encoding(
    session: ReplaySession,
    position: TanniPosition,
    *,
    bin_size_cm: float,
    smoothing_sigma_bins: float,
    running_speed_cm_s: float,
    min_running_spikes: int,
    max_mean_rate_hz: float,
    min_peak_rate_hz: float,
    min_split_half_stability: float,
) -> tuple[EncodingModel, pd.DataFrame]:
    config = EncodingConfig(
        bin_size_cm=bin_size_cm,
        smoothing_sigma_bins=smoothing_sigma_bins,
        min_speed_cm_s=running_speed_cm_s,
        min_occupancy_s=0.02,
        rate_floor_hz=1e-4,
        arena_padding_cm=0.0,
        use_excitatory=True,
        exclude_ripple_intervals=False,
    )
    full = fit_place_field_encoding(session, config)
    midpoint = 0.5 * (float(session.run_times[0, 0]) + float(session.run_times[-1, 1]))
    first = fit_place_field_encoding(replace(session, run_times=np.array([[session.run_times[0, 0], midpoint]])), config)
    second = fit_place_field_encoding(replace(session, run_times=np.array([[midpoint, session.run_times[-1, 1]]])), config)
    valid_position = position.valid & np.isfinite(position.speed_cm_s) & (position.speed_cm_s >= running_speed_cm_s)
    frame_dt = np.diff(position.times_s, append=position.times_s[-1] + np.median(np.diff(position.times_s)))
    running_duration = float(np.sum(frame_dt[valid_position]))
    spike_times = session.spikes[:, 0] if session.spikes.size else np.empty(0)
    spike_ids = session.spikes[:, 1].astype(int) if session.spikes.size else np.empty(0, dtype=int)
    spike_speed = np.interp(spike_times, position.times_s, np.nan_to_num(position.speed_cm_s, nan=0.0)) if spike_times.size else np.empty(0)
    rows: list[dict[str, object]] = []
    selected: list[int] = []
    for row_index, cell_id in enumerate(full.cell_ids.astype(int)):
        running_spikes = int(np.count_nonzero((spike_ids == cell_id) & (spike_speed >= running_speed_cm_s)))
        mean_rate = float(running_spikes / max(running_duration, np.finfo(float).eps))
        peak_rate = float(np.nanmax(full.rates_hz[row_index]))
        occupied = (first.occupancy_s >= config.min_occupancy_s) & (second.occupancy_s >= config.min_occupancy_s)
        stability = _safe_correlation(first.rates_hz[row_index, occupied], second.rates_hz[row_index, occupied])
        decoder_passed = bool(
            running_spikes >= min_running_spikes
            and mean_rate <= max_mean_rate_hz
            and peak_rate >= min_peak_rate_hz
            and np.isfinite(stability)
            and stability >= min_split_half_stability
        )
        paper_place_like = bool(mean_rate <= 4.0 and peak_rate >= 2.0 and np.isfinite(stability) and stability >= 0.25)
        if decoder_passed:
            selected.append(cell_id)
        rows.append(
            {
                "animal": session.rat,
                "session": session.name,
                "cell_id": cell_id,
                "tetrode": cell_id // 1000,
                "cluster_id": cell_id % 1000,
                "running_spikes": running_spikes,
                "mean_running_rate_hz": mean_rate,
                "peak_rate_hz": peak_rate,
                "split_half_stability": stability,
                "decoder_unit_passed": decoder_passed,
                "paper_place_like_without_waveform_gate": paper_place_like,
            }
        )
    if len(selected) < 3:
        raise RuntimeError(f"{session.session_id}: only {len(selected)} decoder units passed predeclared QC")
    return full.select_cells(selected), pd.DataFrame(rows)


def crossvalidated_decoder_qc(
    session: ReplaySession,
    position: TanniPosition,
    encoding: EncodingModel,
    *,
    running_speed_cm_s: float,
    window_s: float,
    n_folds: int,
    samples_per_fold: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = float(session.run_times[0, 0])
    end = float(session.run_times[-1, 1])
    boundaries = np.linspace(start, end, n_folds + 1)
    rows: list[dict[str, object]] = []
    selected_session = replace(session, excitatory_neurons=encoding.cell_ids)
    for fold in range(n_folds):
        test_start = float(boundaries[fold])
        test_end = float(boundaries[fold + 1])
        train_intervals = []
        if test_start > start:
            train_intervals.append([start, test_start])
        if test_end < end:
            train_intervals.append([test_end, end])
        fold_session = replace(selected_session, run_times=np.asarray(train_intervals, dtype=float))
        fold_encoding = fit_place_field_encoding(fold_session, encoding.config).select_cells(encoding.cell_ids)
        candidate = np.flatnonzero(
            position.valid
            & np.isfinite(position.speed_cm_s)
            & (position.speed_cm_s >= running_speed_cm_s)
            & (position.times_s >= test_start + 0.5 * window_s)
            & (position.times_s < test_end - 0.5 * window_s)
        )
        if candidate.size == 0:
            continue
        if candidate.size > samples_per_fold:
            candidate = np.sort(rng.choice(candidate, size=samples_per_fold, replace=False))
        counts = _windowed_spike_counts(
            selected_session.spikes,
            encoding.cell_ids,
            position.times_s[candidate],
            window_s,
        )
        log_likelihood = _poisson_window_log_likelihood(counts, fold_encoding.rates_hz, window_s)
        valid_bins = fold_encoding.occupancy_s >= 0.05
        log_likelihood[:, ~valid_bins] = -np.inf
        posterior = posterior_from_log_likelihood(log_likelihood)
        decoded = posterior @ fold_encoding.bin_centers
        errors = np.linalg.norm(decoded - position.xy_cm[candidate], axis=1)
        wall = nearest_wall_distance(position.xy_cm[candidate], np.asarray(session.metadata["arena_size_cm"], dtype=float))
        for local, position_index in enumerate(candidate):
            rows.append(
                {
                    "animal": session.rat,
                    "session": session.name,
                    "fold": fold,
                    "time_s": float(position.times_s[position_index]),
                    "actual_x_cm": float(position.xy_cm[position_index, 0]),
                    "actual_y_cm": float(position.xy_cm[position_index, 1]),
                    "decoded_x_cm": float(decoded[local, 0]),
                    "decoded_y_cm": float(decoded[local, 1]),
                    "decoder_error_cm": float(errors[local]),
                    "wall_distance_cm": float(wall[local]),
                    "wall_distance_normalized": float(wall[local] / (0.5 * np.min(session.metadata["arena_size_cm"]))),
                    "n_spikes": int(counts[local].sum()),
                    "n_active_cells": int(np.count_nonzero(counts[local])),
                }
            )
    return pd.DataFrame(rows)


def detect_supported_events(
    path: Path,
    session: ReplaySession,
    position: TanniPosition,
    encoding: EncodingModel,
    output_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    cache = output_dir / "ripple-envelope-cache" / f"{session.rat}_{session.name}.npz"
    if args.reuse_ripple_envelope and cache.exists():
        cached = np.load(cache)
        timestamps = cached["timestamps_s"]
        aggregate_z = cached["aggregate_z"]
        sample_rate = float(cached["sample_rate_hz"])
    else:
        timestamps, aggregate_z, sample_rate = aggregate_ripple_envelope_z(path, max_channels=args.max_lfp_channels)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, timestamps_s=timestamps, aggregate_z=aggregate_z, sample_rate_hz=sample_rate)
    candidates = detect_ripple_candidates(
        timestamps,
        aggregate_z,
        threshold_z=args.ripple_threshold_z,
        peak_threshold_z=args.ripple_peak_threshold_z,
        min_duration_s=args.ripple_min_duration_s,
        max_duration_s=args.ripple_max_duration_s,
        merge_gap_s=args.ripple_merge_gap_s,
    )
    selected_spikes = session.spikes[np.isin(session.spikes[:, 1].astype(int), encoding.cell_ids)]
    rows: list[dict[str, object]] = []
    arena = np.asarray(session.metadata["arena_size_cm"], dtype=float)
    for event_index, candidate in enumerate(candidates):
        window_start = candidate.peak_time_s - args.event_half_window_s
        window_end = candidate.peak_time_s + args.event_half_window_s
        position_mask = (position.times_s >= window_start) & (position.times_s <= window_end) & position.valid
        mean_speed = float(np.nanmean(position.speed_cm_s[position_mask])) if np.any(position_mask) else np.nan
        peak_xy = np.column_stack(
            (
                np.interp([candidate.peak_time_s], position.times_s, position.xy_cm[:, 0]),
                np.interp([candidate.peak_time_s], position.times_s, position.xy_cm[:, 1]),
            )
        )
        animal_wall_distance = float(nearest_wall_distance(peak_xy, arena)[0])
        event_spikes = selected_spikes[(selected_spikes[:, 0] >= window_start) & (selected_spikes[:, 0] < window_end)]
        n_spikes = int(event_spikes.shape[0])
        n_active = int(np.unique(event_spikes[:, 1].astype(int)).shape[0]) if n_spikes else 0
        immobile = bool(np.isfinite(mean_speed) and mean_speed < args.immobility_speed_cm_s)
        supported = bool(immobile and n_spikes >= args.min_event_spikes and n_active >= args.min_event_active_cells)
        rows.append(
            {
                "animal": session.rat,
                "session": session.name,
                "event_index": event_index,
                "core_start_time_s": candidate.start_time_s,
                "core_end_time_s": candidate.end_time_s,
                "core_duration_ms": candidate.duration_s * 1000.0,
                "peak_time_s": candidate.peak_time_s,
                "peak_ripple_z": candidate.peak_ripple_z,
                "window_start_time_s": window_start,
                "window_end_time_s": window_end,
                "window_duration_ms": (window_end - window_start) * 1000.0,
                "animal_mean_speed_cm_s": mean_speed,
                "animal_wall_distance_cm": animal_wall_distance,
                "n_spikes": n_spikes,
                "n_active_cells": n_active,
                "immobile": immobile,
                "spike_supported": n_spikes >= args.min_event_spikes and n_active >= args.min_event_active_cells,
                "selected_for_decoding": supported,
                "event_definition": "channelwise_150_250Hz_envelope_then_mean_fixed_peak_window",
            }
        )
    return pd.DataFrame(rows)


def decode_selected_events(
    session: ReplaySession,
    encoding: EncodingModel,
    events: pd.DataFrame,
    *,
    decode_bin_s: float,
    arena_size_cm: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gradients = local_poisson_code_gradient(
        encoding.bin_centers,
        encoding.rates_hz,
        neighbor_radius_cm=1.6 * float(encoding.config.bin_size_cm),
    )
    selected_session = replace(session, excitatory_neurons=encoding.cell_ids)
    event_rows: list[dict[str, object]] = []
    segment_frames: list[pd.DataFrame] = []
    for row in events.loc[events["selected_for_decoding"]].itertuples(index=False):
        ripple = RippleEvent(
            start=float(row.window_start_time_s),
            end=float(row.window_end_time_s),
            peak=float(row.peak_time_s),
            raw_power=float(row.peak_ripple_z),
            z_power_session=float(row.peak_ripple_z),
            z_power_epoch=float(row.peak_ripple_z),
        )
        emissions = build_emissions(selected_session, encoding, ripple, EmissionConfig(time_bin_s=decode_bin_s))
        log_likelihood = emissions.log_likelihood.copy()
        valid_bins = encoding.occupancy_s >= 0.05
        log_likelihood[:, ~valid_bins] = -np.inf
        posterior = posterior_from_log_likelihood(log_likelihood)
        metrics = posterior_path_segments(
            posterior,
            encoding.bin_centers,
            encoding.rates_hz,
            encoding.occupancy_s,
            emissions.times,
            arena_size_cm,
        )
        local_gradient = posterior @ np.nan_to_num(gradients, nan=float(np.nanmedian(gradients)))
        metrics["local_code_gradient_sqrt_hz_per_cm"] = 0.5 * (local_gradient[:-1] + local_gradient[1:])
        frame = pd.DataFrame(metrics)
        frame.insert(0, "segment_index", np.arange(frame.shape[0]))
        frame.insert(0, "event_index", int(row.event_index))
        frame.insert(0, "session", session.name)
        frame.insert(0, "animal", session.rat)
        frame["peak_ripple_z"] = float(row.peak_ripple_z)
        frame["event_n_spikes"] = int(row.n_spikes)
        frame["event_n_active_cells"] = int(row.n_active_cells)
        frame["path_estimator"] = "independent_emission_posterior_mean"
        segment_frames.append(frame)
        posterior_mean = posterior @ encoding.bin_centers
        event_rows.append(
            {
                "animal": session.rat,
                "session": session.name,
                "event_index": int(row.event_index),
                "peak_time_s": float(row.peak_time_s),
                "peak_ripple_z": float(row.peak_ripple_z),
                "n_spikes": int(row.n_spikes),
                "n_active_cells": int(row.n_active_cells),
                "posterior_path_length_cm": float(np.linalg.norm(np.diff(posterior_mean, axis=0), axis=1).sum()),
                "posterior_net_displacement_cm": float(np.linalg.norm(posterior_mean[-1] - posterior_mean[0])),
                "median_physical_speed_cm_s": float(np.median(frame["physical_speed_cm_s"])),
                "median_map_speed_cm_s": float(np.median(frame["map_speed_cm_s"])),
                "median_posterior_rms_independent_speed_cm_s": float(np.median(frame["posterior_rms_independent_speed_cm_s"])),
                "median_code_speed_sqrt_hz_per_s": float(np.median(frame["code_speed_sqrt_hz_per_s"])),
                "median_wall_distance_cm": float(np.median(frame["wall_distance_cm"])),
                "median_posterior_entropy": float(np.median(frame["posterior_entropy"])),
                "median_posterior_spread_cm": float(np.median(frame["posterior_spread_cm"])),
                "decoded_mobile_20cm": bool(np.linalg.norm(np.diff(posterior_mean, axis=0), axis=1).sum() >= 20.0),
            }
        )
    segments = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    return pd.DataFrame(event_rows), segments


def simulate_constant_speed_decoder_null(
    session: ReplaySession,
    encoding: EncodingModel,
    *,
    n_events: int,
    n_time_bins: int,
    dt_s: float,
    true_speed_cm_s: float,
    seed: int,
) -> pd.DataFrame:
    """Decode Poisson spikes generated from constant-speed reflected paths."""

    rng = np.random.default_rng(seed)
    arena = np.asarray(session.metadata["arena_size_cm"], dtype=float)
    centers = encoding.bin_centers
    valid_bins = encoding.occupancy_s >= 0.05
    valid_indices = np.flatnonzero(valid_bins)
    if valid_indices.size == 0:
        return pd.DataFrame()
    wall = nearest_wall_distance(centers[valid_indices], arena) / (0.5 * float(np.min(arena)))
    quartile = np.minimum((wall * 4.0).astype(int), 3)
    events: list[pd.DataFrame] = []
    for event_index in range(n_events):
        target_quartile = event_index % 4
        possible = valid_indices[quartile == target_quartile]
        if possible.size == 0:
            possible = valid_indices
        start_bin = int(rng.choice(possible))
        path = np.empty((n_time_bins, 2), dtype=float)
        path[0] = centers[start_bin]
        angle = float(rng.uniform(-np.pi, np.pi))
        velocity = true_speed_cm_s * np.array([np.cos(angle), np.sin(angle)], dtype=float)
        for time_index in range(1, n_time_bins):
            next_position = path[time_index - 1] + velocity * dt_s
            for dimension in range(2):
                limit = float(arena[dimension])
                while next_position[dimension] < 0.0 or next_position[dimension] > limit:
                    if next_position[dimension] < 0.0:
                        next_position[dimension] = -next_position[dimension]
                        velocity[dimension] *= -1.0
                    if next_position[dimension] > limit:
                        next_position[dimension] = 2.0 * limit - next_position[dimension]
                        velocity[dimension] *= -1.0
            path[time_index] = next_position
        squared_distance = np.sum((path[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        true_bins = np.argmin(squared_distance, axis=1)
        expected_counts = encoding.rates_hz[:, true_bins].T * dt_s
        counts = rng.poisson(expected_counts)
        log_likelihood = _poisson_window_log_likelihood(counts, encoding.rates_hz, dt_s)
        log_likelihood[:, ~valid_bins] = -np.inf
        posterior = posterior_from_log_likelihood(log_likelihood)
        metrics = posterior_path_segments(
            posterior,
            centers,
            encoding.rates_hz,
            encoding.occupancy_s,
            np.arange(n_time_bins, dtype=float) * dt_s,
            arena,
        )
        frame = pd.DataFrame(metrics)
        frame.insert(0, "segment_index", np.arange(frame.shape[0]))
        frame.insert(0, "event_index", event_index)
        frame.insert(0, "session", session.name)
        frame.insert(0, "animal", session.rat)
        true_midpoints = 0.5 * (path[:-1] + path[1:])
        frame["true_wall_distance_cm"] = nearest_wall_distance(true_midpoints, arena)
        frame["true_speed_cm_s"] = np.linalg.norm(np.diff(path, axis=0), axis=1) / dt_s
        frame["simulated_spikes"] = int(counts.sum())
        events.append(frame)
    return pd.concat(events, ignore_index=True) if events else pd.DataFrame()


def summarize_decoder_qc(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (animal, session), frame in samples.groupby(["animal", "session"], sort=True):
        rows.append(
            {
                "animal": animal,
                "session": session,
                "n_crossval_samples": int(frame.shape[0]),
                "median_decoder_error_cm": float(frame["decoder_error_cm"].median()),
                "p75_decoder_error_cm": float(frame["decoder_error_cm"].quantile(0.75)),
                "p90_decoder_error_cm": float(frame["decoder_error_cm"].quantile(0.90)),
                "median_decoder_error_near_wall_cm": _median_where(frame, "decoder_error_cm", frame["wall_distance_normalized"] < 0.25),
                "median_decoder_error_far_wall_cm": _median_where(frame, "decoder_error_cm", frame["wall_distance_normalized"] >= 0.75),
                "finite_decoder_fraction": float(np.isfinite(frame["decoder_error_cm"]).mean()),
                "decoder_qc_available": bool(frame.shape[0] > 0 and np.isfinite(frame["decoder_error_cm"]).all()),
            }
        )
    return pd.DataFrame(rows)


def wall_quartile_summary(segments: pd.DataFrame, decoder_samples: pd.DataFrame) -> pd.DataFrame:
    replay = segments.copy()
    replay["wall_quartile"] = _fixed_wall_quartile(replay["wall_distance_normalized"])
    event_level = (
        replay.groupby(["animal", "session", "event_index", "wall_quartile"], observed=True)
        .agg(
            physical_speed_cm_s=("physical_speed_cm_s", "median"),
            map_speed_cm_s=("map_speed_cm_s", "median"),
            posterior_rms_independent_speed_cm_s=("posterior_rms_independent_speed_cm_s", "median"),
            code_speed_sqrt_hz_per_s=("code_speed_sqrt_hz_per_s", "median"),
            posterior_entropy=("posterior_entropy", "median"),
            posterior_spread_cm=("posterior_spread_cm", "median"),
            local_code_gradient_sqrt_hz_per_cm=("local_code_gradient_sqrt_hz_per_cm", "median"),
            n_event_segments=("segment_index", "size"),
        )
        .reset_index()
    )
    decoder = decoder_samples.copy()
    decoder["wall_quartile"] = _fixed_wall_quartile(decoder["wall_distance_normalized"])
    decoder_by_animal = (
        decoder.groupby(["animal", "wall_quartile"], observed=True)["decoder_error_cm"].median().rename("crossval_decoder_error_cm").reset_index()
    )
    animal = (
        event_level.groupby(["animal", "wall_quartile"], observed=True)
        .agg(
            physical_speed_cm_s=("physical_speed_cm_s", "median"),
            map_speed_cm_s=("map_speed_cm_s", "median"),
            posterior_rms_independent_speed_cm_s=("posterior_rms_independent_speed_cm_s", "median"),
            code_speed_sqrt_hz_per_s=("code_speed_sqrt_hz_per_s", "median"),
            posterior_entropy=("posterior_entropy", "median"),
            posterior_spread_cm=("posterior_spread_cm", "median"),
            local_code_gradient_sqrt_hz_per_cm=("local_code_gradient_sqrt_hz_per_cm", "median"),
            events=("event_index", "nunique"),
        )
        .reset_index()
        .merge(decoder_by_animal, on=["animal", "wall_quartile"], how="left")
    )
    animal["aggregation_level"] = "animal"
    overall = (
        animal.groupby("wall_quartile", observed=True)
        .agg(
            physical_speed_cm_s=("physical_speed_cm_s", "median"),
            map_speed_cm_s=("map_speed_cm_s", "median"),
            posterior_rms_independent_speed_cm_s=("posterior_rms_independent_speed_cm_s", "median"),
            code_speed_sqrt_hz_per_s=("code_speed_sqrt_hz_per_s", "median"),
            posterior_entropy=("posterior_entropy", "median"),
            posterior_spread_cm=("posterior_spread_cm", "median"),
            local_code_gradient_sqrt_hz_per_cm=("local_code_gradient_sqrt_hz_per_cm", "median"),
            crossval_decoder_error_cm=("crossval_decoder_error_cm", "median"),
            events=("events", "sum"),
        )
        .reset_index()
    )
    overall["animal"] = "ALL_ANIMAL_MEDIAN"
    overall["aggregation_level"] = "animal_balanced_median"
    return pd.concat((animal, overall), ignore_index=True)


def association_summary(
    segments: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    targets = {
        "physical_speed_cm_s": "log1p",
        "map_speed_cm_s": "log1p",
        "posterior_rms_independent_speed_cm_s": "log1p",
        "code_speed_sqrt_hz_per_s": "log1p",
        "local_code_gradient_sqrt_hz_per_cm": "log1p",
        "posterior_entropy": "identity",
        "posterior_spread_cm": "log1p",
    }
    controls = [
        "posterior_entropy",
        "posterior_spread_cm",
        "local_code_gradient_sqrt_hz_per_cm",
        "local_occupancy_s",
        "event_n_spikes",
        "event_n_active_cells",
        "peak_ripple_z",
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    event_segments = (
        segments.groupby(["animal", "session", "event_index"], as_index=False)
        .agg(
            wall_distance_normalized=("wall_distance_normalized", "median"),
            physical_speed_cm_s=("physical_speed_cm_s", "median"),
            map_speed_cm_s=("map_speed_cm_s", "median"),
            posterior_rms_independent_speed_cm_s=("posterior_rms_independent_speed_cm_s", "median"),
            code_speed_sqrt_hz_per_s=("code_speed_sqrt_hz_per_s", "median"),
            local_code_gradient_sqrt_hz_per_cm=("local_code_gradient_sqrt_hz_per_cm", "median"),
            posterior_entropy=("posterior_entropy", "median"),
            posterior_spread_cm=("posterior_spread_cm", "median"),
            local_occupancy_s=("local_occupancy_s", "median"),
            event_n_spikes=("event_n_spikes", "first"),
            event_n_active_cells=("event_n_active_cells", "first"),
            peak_ripple_z=("peak_ripple_z", "first"),
        )
    )
    for aggregation_name, analysis_frame in (("event", event_segments), ("segment", segments)):
        animal_scope = "animal" if aggregation_name == "event" else "animal_segment_sensitivity"
        aggregate_scope = "animal_balanced" if aggregation_name == "event" else "segment_animal_balanced_sensitivity"
        count_name = "n_events" if aggregation_name == "event" else "n_segments"
        for target, transform in targets.items():
            per_animal_raw: list[float] = []
            per_animal_adjusted: list[float] = []
            for animal, frame in analysis_frame.groupby("animal", sort=True):
                raw = _spearman(frame["wall_distance_normalized"], frame[target])
                adjusted = _partial_rank_correlation(frame, "wall_distance_normalized", target, [name for name in controls if name != target])
                per_animal_raw.append(raw)
                per_animal_adjusted.append(adjusted)
                rows.append(
                    {
                        "metric": target,
                        "scope": animal_scope,
                        "animal": animal,
                        count_name: int(frame.shape[0]),
                        "raw_spearman_r": raw,
                        "quality_adjusted_partial_r": adjusted,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "transform": transform,
                    }
                )
            raw_ci = _animal_bootstrap_ci(np.asarray(per_animal_raw), bootstrap_replicates, rng)
            adjusted_ci = _animal_bootstrap_ci(np.asarray(per_animal_adjusted), bootstrap_replicates, rng)
            rows.append(
                {
                    "metric": target,
                    "scope": aggregate_scope,
                    "animal": "ALL",
                    count_name: int(analysis_frame.shape[0]),
                    "raw_spearman_r": _nanmedian_or_nan(per_animal_raw),
                    "quality_adjusted_partial_r": _nanmedian_or_nan(per_animal_adjusted),
                    "ci95_low": adjusted_ci[0],
                    "ci95_high": adjusted_ci[1],
                    "raw_ci95_low": raw_ci[0],
                    "raw_ci95_high": raw_ci[1],
                    "positive_animals_raw": int(np.count_nonzero(np.asarray(per_animal_raw) > 0.0)),
                    "positive_animals_adjusted": int(np.count_nonzero(np.asarray(per_animal_adjusted) > 0.0)),
                    "animals": int(len(per_animal_raw)),
                    "transform": transform,
                }
            )
    if not synthetic.empty:
        synthetic_events = (
            synthetic.groupby(["animal", "session", "event_index"], as_index=False)
            .agg(
                true_wall_distance_cm=("true_wall_distance_cm", "median"),
                decoded_wall_distance_normalized=("wall_distance_normalized", "median"),
                physical_speed_cm_s=("physical_speed_cm_s", "median"),
            )
        )
        for metric, wall_column in (
            ("synthetic_decoded_wall_physical_speed_cm_s", "decoded_wall_distance_normalized"),
            ("synthetic_true_wall_physical_speed_cm_s", "true_wall_distance_cm"),
        ):
            null_values = []
            for animal, frame in synthetic_events.groupby("animal", sort=True):
                rho = _spearman(frame[wall_column], frame["physical_speed_cm_s"])
                null_values.append(rho)
                rows.append(
                    {
                        "metric": metric,
                        "scope": "animal",
                        "animal": animal,
                        "n_events": int(frame.shape[0]),
                        "raw_spearman_r": rho,
                        "quality_adjusted_partial_r": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "transform": "constant_true_physical_speed",
                    }
                )
            null_ci = _animal_bootstrap_ci(np.asarray(null_values), bootstrap_replicates, rng)
            rows.append(
                {
                    "metric": metric,
                    "scope": "animal_balanced",
                    "animal": "ALL",
                    "n_events": int(synthetic_events.shape[0]),
                    "raw_spearman_r": _nanmedian_or_nan(null_values),
                    "quality_adjusted_partial_r": np.nan,
                    "ci95_low": null_ci[0],
                    "ci95_high": null_ci[1],
                    "positive_animals_raw": int(np.count_nonzero(np.asarray(null_values) > 0.0)),
                    "animals": int(len(null_values)),
                    "transform": "constant_true_physical_speed",
                }
            )
    return pd.DataFrame(rows)


def build_gate_summary(
    manifest: pd.DataFrame,
    unit_qc: pd.DataFrame,
    decoder_summary: pd.DataFrame,
    ripple_events: pd.DataFrame,
    event_speed: pd.DataFrame,
    segments: pd.DataFrame,
    associations: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    n_animals = int(manifest["animal"].nunique())
    selected = ripple_events.loc[ripple_events["selected_for_decoding"]]
    decoder_animals = int(decoder_summary.loc[decoder_summary["decoder_qc_available"], "animal"].nunique()) if not decoder_summary.empty else 0
    selected_per_animal = selected.groupby("animal").size() if not selected.empty else pd.Series(dtype=int)
    technical = {
        "five_large_arena_animals_present": (n_animals == 5, f"{n_animals}/5"),
        "decoder_units_present_all_animals": (
            int(unit_qc.loc[unit_qc["decoder_unit_passed"], "animal"].nunique()) == n_animals,
            f"{unit_qc.loc[unit_qc['decoder_unit_passed'], 'animal'].nunique()}/{n_animals}",
        ),
        "heldout_run_decoder_available_all_animals": (decoder_animals == n_animals, f"{decoder_animals}/{n_animals}"),
        "ripple_candidates_selected_all_animals": (
            int(selected["animal"].nunique()) == n_animals if not selected.empty else False,
            f"{selected['animal'].nunique() if not selected.empty else 0}/{n_animals}",
        ),
        "at_least_20_selected_events_per_animal": (
            bool(not selected_per_animal.empty and (selected_per_animal >= 20).all()),
            ";".join(f"{animal}:{count}" for animal, count in selected_per_animal.items()),
        ),
        "decoded_segments_present": (not segments.empty, str(int(segments.shape[0]))),
        "constant_speed_decoder_null_present": (
            bool((associations["metric"] == "synthetic_decoded_wall_physical_speed_cm_s").any()),
            "present" if (associations["metric"] == "synthetic_decoded_wall_physical_speed_cm_s").any() else "missing",
        ),
    }
    overall_technical = all(passed for passed, _observed in technical.values())
    physical = _association_row(associations, "physical_speed_cm_s")
    code = _association_row(associations, "code_speed_sqrt_hz_per_s")
    null = _association_row(associations, "synthetic_decoded_wall_physical_speed_cm_s")
    adjusted_excludes_zero = bool(np.isfinite(physical.get("ci95_low", np.nan)) and (physical["ci95_low"] > 0.0 or physical["ci95_high"] < 0.0))
    raw_uniform = int(physical.get("positive_animals_raw", 0)) in {0, n_animals}
    observed_outside_null = bool(
        np.isfinite(null.get("ci95_low", np.nan))
        and (physical["raw_spearman_r"] < null["ci95_low"] or physical["raw_spearman_r"] > null["ci95_high"])
    )
    if not overall_technical:
        verdict = "technical_incomplete"
    elif adjusted_excludes_zero and raw_uniform and observed_outside_null:
        verdict = "wall_distance_association_survives_decoder_controls"
    elif abs(float(physical.get("raw_spearman_r", np.nan))) > 0.05 and not adjusted_excludes_zero:
        verdict = "raw_wall_distance_association_decoder_mediated_or_inconclusive"
    else:
        verdict = "no_robust_wall_distance_speed_association"
    rows = [
        {"gate": name, "passed": passed, "observed": observed, "criterion": name.replace("_", " ")}
        for name, (passed, observed) in technical.items()
    ]
    rows.extend(
        [
            {"gate": "overall_technical", "passed": overall_technical, "observed": f"{sum(p for p, _ in technical.values())}/{len(technical)}", "criterion": "all technical gates pass"},
            {
                "gate": "physical_speed_adjusted_effect_excludes_zero",
                "passed": adjusted_excludes_zero,
                "observed": f"rho={physical.get('quality_adjusted_partial_r', np.nan):.4f}; CI=[{physical.get('ci95_low', np.nan):.4f},{physical.get('ci95_high', np.nan):.4f}]",
                "criterion": "animal-balanced quality-adjusted 95% bootstrap CI excludes zero",
            },
            {
                "gate": "physical_speed_effect_outside_constant_speed_decoder_null",
                "passed": observed_outside_null,
                "observed": f"observed={physical.get('raw_spearman_r', np.nan):.4f}; null_CI=[{null.get('ci95_low', np.nan):.4f},{null.get('ci95_high', np.nan):.4f}]",
                "criterion": "observed animal-median rho lies outside constant-physical-speed decoder-null CI",
            },
            {
                "gate": "code_speed_wall_association_descriptive",
                "passed": None,
                "observed": f"raw={code.get('raw_spearman_r', np.nan):.4f}; adjusted={code.get('quality_adjusted_partial_r', np.nan):.4f}",
                "criterion": "descriptive; distinguishes physical and representational speed",
            },
            {"gate": "biological_verdict", "passed": verdict == "wall_distance_association_survives_decoder_controls", "observed": verdict, "criterion": "predeclared conservative interpretation"},
        ]
    )
    return pd.DataFrame(rows), verdict


def make_figure(
    quartiles: pd.DataFrame,
    associations: pd.DataFrame,
    output_path: Path,
) -> None:
    animal = quartiles.loc[quartiles["aggregation_level"] == "animal"].copy()
    overall = quartiles.loc[quartiles["aggregation_level"] == "animal_balanced_median"].copy()
    order = ["Q1_nearest", "Q2", "Q3", "Q4_farthest"]
    x = np.arange(4)
    event_counts = overall.set_index("wall_quartile")["events"].reindex(order).fillna(0).astype(int)
    x_labels = [f"Q1\nnear wall\nn={event_counts.iloc[0]}", f"Q2\nn={event_counts.iloc[1]}", f"Q3\nn={event_counts.iloc[2]}", f"Q4\nfar wall\nn={event_counts.iloc[3]}"]
    panels = [
        ("physical_speed_cm_s", "Physical posterior-mean speed", "cm/s"),
        ("code_speed_sqrt_hz_per_s", "Population-code speed", "sqrt(Hz)/s"),
        ("local_code_gradient_sqrt_hz_per_cm", "Local code resolution", "sqrt(Hz)/cm"),
        ("crossval_decoder_error_cm", "Held-out RUN decoder error", "cm"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (column, title, ylabel) in zip(axes.ravel(), panels, strict=True):
        for _animal, frame in animal.groupby("animal", sort=True):
            values = frame.set_index("wall_quartile")[column].reindex(order)
            axis.plot(x, values, color="#7c8794", alpha=0.45, linewidth=1.2, marker="o", markersize=3)
        values = overall.set_index("wall_quartile")[column].reindex(order)
        axis.plot(x, values, color="#bb2e3f", linewidth=2.6, marker="o", markersize=6, label="animal-balanced median")
        axis.set_xticks(x, x_labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", color="#d9dde2", linewidth=0.7)
    physical = _association_row(associations, "physical_speed_cm_s")
    code = _association_row(associations, "code_speed_sqrt_hz_per_s")
    null = _association_row(associations, "synthetic_decoded_wall_physical_speed_cm_s")
    axes[0, 0].text(
        0.02,
        0.98,
        f"raw animal-median rho = {physical.get('raw_spearman_r', np.nan):+.2f}\nadjusted rho = {physical.get('quality_adjusted_partial_r', np.nan):+.2f}\nconstant-speed null rho = {null.get('raw_spearman_r', np.nan):+.2f}",
        transform=axes[0, 0].transAxes,
        va="top",
        fontsize=9,
    )
    axes[0, 1].text(
        0.02,
        0.98,
        f"raw animal-median rho = {code.get('raw_spearman_r', np.nan):+.2f}\nadjusted rho = {code.get('quality_adjusted_partial_r', np.nan):+.2f}",
        transform=axes[0, 1].transAxes,
        va="top",
        fontsize=9,
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Tanni et al. large 2D arenas: awake ripple-associated decoded speed", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(
    output_path: Path,
    manifest: pd.DataFrame,
    unit_qc: pd.DataFrame,
    decoder_summary: pd.DataFrame,
    ripple_events: pd.DataFrame,
    event_speed: pd.DataFrame,
    associations: pd.DataFrame,
    verdict: str,
) -> None:
    selected = ripple_events.loc[ripple_events["selected_for_decoding"]]
    physical = _association_row(associations, "physical_speed_cm_s")
    map_speed = _association_row(associations, "map_speed_cm_s")
    rms_speed = _association_row(associations, "posterior_rms_independent_speed_cm_s")
    code = _association_row(associations, "code_speed_sqrt_hz_per_s")
    gradient = _association_row(associations, "local_code_gradient_sqrt_hz_per_cm")
    null = _association_row(associations, "synthetic_decoded_wall_physical_speed_cm_s")
    lines = [
        "# Tanni et al. large-2D wall-distance replay-speed test",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "## Scope and claim boundary",
        "",
        "This run tests awake, immobile, ripple-band-positive, spike-supported candidate events in the largest (350 x 250 cm) arena. "
        "The primary path is the independent-bin emission posterior mean, so a motion prior cannot manufacture the wall-speed relation. "
        "Wall distance is the posterior expectation of binwise wall distance, not the distance of the posterior mean; this avoids assigning diffuse posteriors to the arena center by construction. "
        "Events are ripple-associated candidate sequences until model-evidence classification is joined; this report does not call every event replay.",
        "",
        "## Technical coverage",
        "",
        f"- Sessions / animals: {manifest.shape[0]} / {manifest['animal'].nunique()}",
        f"- Decoder-QC units: {int(unit_qc['decoder_unit_passed'].sum())}",
        f"- Held-out RUN median decoder error across sessions: {decoder_summary['median_decoder_error_cm'].median():.2f} cm",
        f"- Spectral candidates / selected immobile spike-supported events: {ripple_events.shape[0]} / {selected.shape[0]}",
        f"- Decoded events: {event_speed.shape[0]}",
        "",
        "## Wall-distance associations",
        "",
        f"- Physical speed: raw animal-median Spearman rho {physical.get('raw_spearman_r', np.nan):+.3f}; "
        f"quality-adjusted partial rho {physical.get('quality_adjusted_partial_r', np.nan):+.3f}, "
        f"95% animal bootstrap CI [{physical.get('ci95_low', np.nan):+.3f}, {physical.get('ci95_high', np.nan):+.3f}].",
        f"- Population-code speed: raw rho {code.get('raw_spearman_r', np.nan):+.3f}; adjusted rho {code.get('quality_adjusted_partial_r', np.nan):+.3f}.",
        f"- MAP sensitivity: raw rho {map_speed.get('raw_spearman_r', np.nan):+.3f}; adjusted rho {map_speed.get('quality_adjusted_partial_r', np.nan):+.3f}.",
        f"- Independent-posterior RMS sensitivity: raw rho {rms_speed.get('raw_spearman_r', np.nan):+.3f}; adjusted rho {rms_speed.get('quality_adjusted_partial_r', np.nan):+.3f}.",
        f"- Local code gradient: raw rho {gradient.get('raw_spearman_r', np.nan):+.3f}; adjusted rho {gradient.get('quality_adjusted_partial_r', np.nan):+.3f}.",
        f"- Constant-physical-speed decoder null: animal-median decoded-speed rho {null.get('raw_spearman_r', np.nan):+.3f}, "
        f"95% CI [{null.get('ci95_low', np.nan):+.3f}, {null.get('ci95_high', np.nan):+.3f}].",
        "",
        "## Interpretation rule",
        "",
        "A physical-speed gradient is treated as replay-dynamics evidence only when it survives posterior entropy/spread, local code gradient, occupancy, event strength, and ripple-power controls, "
        "is directionally consistent across animals, and lies outside the constant-physical-speed decoder null. Otherwise it is reported as decoder-mediated or inconclusive.",
        "",
        "The original paper established wall-dependent place-code resolution; it did not test replay speed. This analysis therefore reports physical and code-space speed together rather than interpreting raw cm/s alone.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    nwb_paths = sorted(args.dataset_root.resolve().glob(args.nwb_glob))
    if not nwb_paths:
        raise FileNotFoundError(f"No NWB files matched {args.dataset_root / args.nwb_glob}")
    manifest_rows: list[dict[str, object]] = []
    unit_frames: list[pd.DataFrame] = []
    decoder_frames: list[pd.DataFrame] = []
    ripple_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    segment_frames: list[pd.DataFrame] = []
    synthetic_frames: list[pd.DataFrame] = []
    for session_index, nwb_path in enumerate(nwb_paths):
        metadata = read_tanni_session_metadata(nwb_path)
        position = read_tanni_position(nwb_path)
        session = make_replay_session(nwb_path, position)
        encoding, unit_qc = fit_decoder_encoding(
            session,
            position,
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.rate_smoothing_sigma_bins,
            running_speed_cm_s=args.running_speed_cm_s,
            min_running_spikes=args.min_running_spikes,
            max_mean_rate_hz=args.max_mean_rate_hz,
            min_peak_rate_hz=args.min_peak_rate_hz,
            min_split_half_stability=args.min_split_half_stability,
        )
        session = replace(session, excitatory_neurons=encoding.cell_ids)
        decoder = crossvalidated_decoder_qc(
            session,
            position,
            encoding,
            running_speed_cm_s=args.running_speed_cm_s,
            window_s=args.decoder_window_s,
            n_folds=args.decoder_folds,
            samples_per_fold=args.decoder_samples_per_fold,
            seed=args.seed + session_index,
        )
        events = detect_supported_events(nwb_path, session, position, encoding, output_dir, args)
        event_speed, segments = decode_selected_events(
            session,
            encoding,
            events,
            decode_bin_s=args.decode_bin_s,
            arena_size_cm=metadata.arena_size_cm,
        )
        synthetic = simulate_constant_speed_decoder_null(
            session,
            encoding,
            n_events=args.synthetic_events_per_session,
            n_time_bins=max(int(round(2.0 * args.event_half_window_s / args.decode_bin_s)), 3),
            dt_s=args.decode_bin_s,
            true_speed_cm_s=args.synthetic_speed_cm_s,
            seed=args.seed + 1000 + session_index,
        )
        manifest_rows.append(
            {
                "animal": metadata.animal,
                "session": metadata.session,
                "nwb_path": str(nwb_path.resolve()),
                "arena_width_cm": float(metadata.arena_size_cm[0]),
                "arena_height_cm": float(metadata.arena_size_cm[1]),
                "position_samples": int(position.times_s.shape[0]),
                "position_valid_fraction": float(position.valid.mean()),
                "session_duration_s": float(position.times_s[-1] - position.times_s[0]),
                "lfp_sample_rate_hz": metadata.lfp_sample_rate_hz,
                "lfp_channels": metadata.n_lfp_channels,
                "sorted_cells": int(np.unique(session.spikes[:, 1]).shape[0]),
                "decoder_cells": int(encoding.n_cells),
                "spectral_candidates": int(events.shape[0]),
                "selected_events": int(events["selected_for_decoding"].sum()),
            }
        )
        unit_frames.append(unit_qc)
        decoder_frames.append(decoder)
        ripple_frames.append(events)
        event_frames.append(event_speed)
        segment_frames.append(segments)
        synthetic_frames.append(synthetic)
        print(
            f"[{session_index + 1}/{len(nwb_paths)}] {metadata.animal}/{metadata.session}: "
            f"cells={encoding.n_cells}, selected_events={int(events['selected_for_decoding'].sum())}, segments={segments.shape[0]}",
            flush=True,
        )
    manifest = pd.DataFrame(manifest_rows)
    unit_qc = pd.concat(unit_frames, ignore_index=True)
    decoder_samples = pd.concat(decoder_frames, ignore_index=True)
    decoder_summary = summarize_decoder_qc(decoder_samples)
    ripple_events = pd.concat(ripple_frames, ignore_index=True)
    event_speed = pd.concat(event_frames, ignore_index=True)
    segments = pd.concat(segment_frames, ignore_index=True)
    synthetic = pd.concat(synthetic_frames, ignore_index=True)
    quartiles = wall_quartile_summary(segments, decoder_samples)
    associations = association_summary(segments, synthetic, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    gates, verdict = build_gate_summary(manifest, unit_qc, decoder_summary, ripple_events, event_speed, segments, associations)
    tables = {
        "tanni2022_session_manifest.csv": manifest,
        "tanni2022_unit_qc.csv": unit_qc,
        "tanni2022_decoder_qc_samples.csv": decoder_samples,
        "tanni2022_decoder_qc_summary.csv": decoder_summary,
        "tanni2022_ripple_candidates.csv": ripple_events,
        "tanni2022_replay_speed_events.csv": event_speed,
        "tanni2022_replay_speed_segments.csv": segments,
        "tanni2022_wall_distance_quartiles.csv": quartiles,
        "tanni2022_wall_distance_associations.csv": associations,
        "tanni2022_synthetic_constant_speed_null.csv": synthetic,
        "tanni2022_wall_distance_gate_summary.csv": gates,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    make_figure(quartiles, associations, output_dir / "tanni2022_wall_distance_replay_figure.png")
    write_summary(
        output_dir / "tanni2022_wall_distance_replay_summary.md",
        manifest,
        unit_qc,
        decoder_summary,
        ripple_events,
        event_speed,
        associations,
        verdict,
    )
    provenance = build_script_provenance(input_paths={f"nwb_{index}": path for index, path in enumerate(nwb_paths)}, argv=sys.argv)
    manifest_json = {
        "dataset": "tanni_de_cothi_barry_2022_large_2d",
        "dataset_doi": DATASET_DOI,
        "paper_doi": PAPER_DOI,
        "archive_url": ARCHIVE_URL,
        "archive_md5": ARCHIVE_MD5,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "event_definition": "channelwise 150-250 Hz Hilbert-envelope robust z, mean after envelope extraction, fixed peak-centered decoding window",
        "path_estimator": "independent emission posterior mean",
        "verdict": verdict,
        "outputs": list(REQUIRED_OUTPUTS),
        "provenance": provenance,
    }
    (output_dir / "tanni2022_wall_distance_manifest.json").write_text(json.dumps(manifest_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Required outputs were not written: {missing}")
    return 0


def _windowed_spike_counts(spikes: np.ndarray, cell_ids: np.ndarray, centers_s: np.ndarray, window_s: float) -> np.ndarray:
    counts = np.zeros((centers_s.shape[0], cell_ids.shape[0]), dtype=int)
    half = 0.5 * float(window_s)
    for column, cell_id in enumerate(cell_ids.astype(int)):
        times = spikes[spikes[:, 1].astype(int) == cell_id, 0]
        left = np.searchsorted(times, centers_s - half, side="left")
        right = np.searchsorted(times, centers_s + half, side="left")
        counts[:, column] = right - left
    return counts


def _poisson_window_log_likelihood(counts: np.ndarray, rates_hz: np.ndarray, duration_s: float) -> np.ndarray:
    expected = np.maximum(np.asarray(rates_hz, dtype=float) * float(duration_s), np.finfo(float).tiny)
    values = np.asarray(counts, dtype=float)
    return values @ np.log(expected) - expected.sum(axis=0)[None, :] - gammaln(values + 1.0).sum(axis=1)[:, None]


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 6 or np.std(x[valid]) == 0.0 or np.std(y[valid]) == 0.0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _spearman(first: pd.Series | np.ndarray, second: pd.Series | np.ndarray) -> float:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 8 or np.unique(x[valid]).shape[0] < 2 or np.unique(y[valid]).shape[0] < 2:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def _partial_rank_correlation(frame: pd.DataFrame, x_name: str, y_name: str, controls: list[str]) -> float:
    columns = [x_name, y_name, *controls]
    values = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if values.shape[0] < max(20, len(controls) + 5):
        return np.nan
    ranked = values.rank(method="average").to_numpy(dtype=float)
    x = ranked[:, 0]
    y = ranked[:, 1]
    nuisance = ranked[:, 2:]
    nuisance = np.column_stack((np.ones(nuisance.shape[0]), _standardize_columns(nuisance)))
    x_residual = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
    y_residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    if np.std(x_residual) == 0.0 or np.std(y_residual) == 0.0:
        return np.nan
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std[std == 0.0] = 1.0
    return (values - mean) / std


def _animal_bootstrap_ci(values: np.ndarray, replicates: int, rng: np.random.Generator) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    draws = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        draws[index] = float(np.median(rng.choice(finite, size=finite.size, replace=True)))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _nanmedian_or_nan(values: Iterable[float]) -> float:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else np.nan


def _fixed_wall_quartile(values: pd.Series) -> pd.Categorical:
    clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    labels = np.array(["Q1_nearest", "Q2", "Q3", "Q4_farthest"], dtype=object)
    indices = np.minimum(np.floor(clipped * 4.0).astype(int), 3)
    return pd.Categorical(labels[indices], categories=labels.tolist(), ordered=True)


def _median_where(frame: pd.DataFrame, column: str, mask: pd.Series) -> float:
    values = frame.loc[mask, column]
    return float(values.median()) if not values.empty else np.nan


def _association_row(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    rows = frame.loc[(frame["metric"] == metric) & (frame["scope"] == "animal_balanced")]
    return rows.iloc[0].to_dict() if not rows.empty else {}


if __name__ == "__main__":
    raise SystemExit(main())
