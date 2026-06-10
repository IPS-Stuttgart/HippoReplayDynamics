#!/usr/bin/env python3
"""Score spike-matched off-SWR null windows for replay evidence controls."""

from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from benchmark_model_evidence import _check_session, _events, _postprocess_evidence_scores, _session_path
from benchmark_model_evidence_improved import (
    DEFAULT_IMPROVED_STATE_SPACE_IMM_SWITCH_TAU_S,
    DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_PREDICTED_CANDIDATE_TOP_K,
    _family,
    _models,
    _run_settings,
)
from aggregate_event_window_sensitivity import (
    DEFAULT_MARGIN_THRESHOLD,
)
from hipporeplayimm.clusterless import (
    ClusterlessStateSpaceReplayModel,
    build_clusterless_mark_emissions,
    fit_clusterless_mark_encoding,
)
from hipporeplayimm.data import ReplaySession, load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, fit_place_field_encoding
from hipporeplayimm.position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
)
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
    score_replay_model_compat,
)

DEFAULT_MATCHED_NULL_MODELS = (
    "sorted-spike-state-space-stationary "
    "sorted-spike-state-space-diffusion "
    "sorted-spike-state-space-fragmented "
    "sorted-spike-state-space-first-order-imm "
    "sorted-spike-state-space-momentum-exact-sparse "
    "sorted-spike-state-space-momentum "
    "sorted-spike-state-space-imm"
)
DEFAULT_MAX_NON_RUN_CANDIDATE_WINDOWS = 5000

FULL_CORE_REQUIRED_MODELS = (
    "sorted-spike-state-space-stationary",
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
)

FULL_CORE_TRAJECTORY_MODELS = (
    "sorted-spike-state-space-diffusion",
    "sorted-spike-state-space-fragmented",
    "sorted-spike-state-space-first-order-imm",
    "sorted-spike-state-space-momentum-exact-sparse",
)

LIGHTWEIGHT_FO_IMM_STATIONARY_REQUIRED_MODELS = (
    "sorted-spike-state-space-stationary",
    "sorted-spike-state-space-first-order-imm",
)

LIGHTWEIGHT_FO_IMM_STATIONARY_TRAJECTORY_MODELS = (
    "sorted-spike-state-space-first-order-imm",
)

LIGHTWEIGHT_FO_IMM_STATIONARY_SCOPE = "lightweight-first-order-imm-vs-stationary"


def resolve_family_model_sets(
    *,
    comparison_scope: str,
    scored_models: tuple[str, ...] | list[str] | set[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return required and trajectory models for matched-null family margins."""

    scope = str(comparison_scope).strip().lower()

    if scope in {"full-core", "full_core"}:
        return FULL_CORE_REQUIRED_MODELS, FULL_CORE_TRAJECTORY_MODELS

    if scope in {
        LIGHTWEIGHT_FO_IMM_STATIONARY_SCOPE,
        "lightweight_fo_imm_stationary",
        "fo-imm-vs-stationary",
    }:
        return (
            LIGHTWEIGHT_FO_IMM_STATIONARY_REQUIRED_MODELS,
            LIGHTWEIGHT_FO_IMM_STATIONARY_TRAJECTORY_MODELS,
        )

    if scope in {"auto", "from-models"}:
        if scored_models is None:
            raise ValueError("scored_models is required when comparison_scope='auto'")
        scored = set(str(model) for model in scored_models)
        if set(LIGHTWEIGHT_FO_IMM_STATIONARY_REQUIRED_MODELS).issubset(scored) and not set(
            FULL_CORE_REQUIRED_MODELS
        ).issubset(scored):
            return (
                LIGHTWEIGHT_FO_IMM_STATIONARY_REQUIRED_MODELS,
                LIGHTWEIGHT_FO_IMM_STATIONARY_TRAJECTORY_MODELS,
            )
        return FULL_CORE_REQUIRED_MODELS, FULL_CORE_TRAJECTORY_MODELS

    raise ValueError(f"unknown comparison_scope: {comparison_scope!r}")


_FALSE_BOOL_STRINGS = {"", "0", "0.0", "false", "f", "no", "n", "off", "nan", "none", "null"}
_TRUE_BOOL_STRINGS = {"1", "1.0", "true", "t", "yes", "y", "on"}


def _coerce_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
    """Coerce bool-like scalar values without treating all strings as true.

    Pandas ``Series.astype(bool)`` treats every non-empty object string as
    ``True``.  That is unsafe for score CSVs, where ``False``/``0`` may arrive as
    strings after concatenating artifacts.  Keep unknown or missing values on the
    conservative default side so non-comparable evidence rows are not admitted
    into paper-facing family-margin decisions by accident.
    """

    def coerce(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        try:
            if pd.isna(value):
                return bool(default)
        except (TypeError, ValueError):
            return bool(default)
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            return bool(np.isfinite(numeric) and numeric != 0.0)
        text = str(value).strip().lower()
        if text in _TRUE_BOOL_STRINGS:
            return True
        if text in _FALSE_BOOL_STRINGS:
            return False
        try:
            numeric = float(text)
        except ValueError:
            return bool(default)
        return bool(np.isfinite(numeric) and numeric != 0.0)

    return values.map(coerce).astype(bool)


def spike_matched_null_windows(
    session: ReplaySession,
    event_index: int,
    *,
    nulls_per_event: int,
    random_seed: int,
    spike_count_tolerance_fraction: float = 0.10,
    active_cell_tolerance: int | None = None,
    candidate_step_s: float | None = None,
    exclusion_padding_s: float = 0.0,
    restrict_to_run_times: bool = True,
    max_candidate_windows: int | None = DEFAULT_MAX_NON_RUN_CANDIDATE_WINDOWS,
) -> pd.DataFrame:
    """Return off-SWR windows matched to one replay event's spike load."""

    event = session.ripple(int(event_index))
    duration = float(event.end) - float(event.start)
    if duration <= 0.0:
        raise ValueError(f"event {event_index} has non-positive duration")
    spikes = _time_sorted_spikes(session.excitatory_spikes())
    real_count, real_active = _spike_count_and_active_cells(
        spikes,
        float(event.start),
        float(event.end),
        assume_sorted=True,
    )
    candidates = _candidate_null_windows(
        session,
        duration_s=duration,
        candidate_step_s=candidate_step_s,
        exclusion_padding_s=exclusion_padding_s,
        restrict_to_run_times=restrict_to_run_times,
        max_candidate_windows=max_candidate_windows,
        random_seed=int(random_seed) + 104729 * int(event_index),
    )
    if candidates.empty:
        return candidates
    counts: list[int] = []
    active_counts: list[int] = []
    for row in candidates.itertuples(index=False):
        count, active = _spike_count_and_active_cells(
            spikes,
            float(row.window_start_s),
            float(row.window_end_s),
            assume_sorted=True,
        )
        counts.append(count)
        active_counts.append(active)
    candidates = candidates.copy()
    candidates["null_n_spikes"] = counts
    candidates["null_active_cell_count"] = active_counts
    candidates["real_n_spikes"] = int(real_count)
    candidates["real_active_cell_count"] = int(real_active)
    candidates["n_spikes_delta"] = candidates["null_n_spikes"].astype(int) - int(real_count)
    candidates["active_cell_count_delta"] = candidates["null_active_cell_count"].astype(int) - int(real_active)
    denominator = max(int(real_count), 1)
    candidates["n_spikes_relative_delta"] = candidates["n_spikes_delta"].astype(float) / float(denominator)
    within_spike_tolerance = (
        candidates["n_spikes_delta"].abs()
        <= max(1.0, float(spike_count_tolerance_fraction) * float(denominator))
    )
    if active_cell_tolerance is not None:
        within_active_tolerance = candidates["active_cell_count_delta"].abs() <= int(active_cell_tolerance)
    else:
        within_active_tolerance = pd.Series(True, index=candidates.index)
    eligible = candidates[within_spike_tolerance & within_active_tolerance].copy()
    if len(eligible) < int(nulls_per_event):
        eligible = candidates.copy()
    rng = np.random.default_rng(int(random_seed) + 7919 * int(event_index))
    eligible["random_tiebreaker"] = rng.random(len(eligible))
    eligible["n_spikes_delta_abs"] = eligible["n_spikes_delta"].abs()
    eligible["active_cell_count_delta_abs"] = eligible["active_cell_count_delta"].abs()
    eligible = eligible.sort_values(
        ["n_spikes_delta_abs", "active_cell_count_delta_abs", "random_tiebreaker", "window_start_s"],
        kind="mergesort",
    )
    selected = eligible.head(int(nulls_per_event)).copy()
    selected["null_index"] = np.arange(len(selected), dtype=int)
    selected["matched_null_rank"] = selected["null_index"].astype(int) + 1
    selected["template_event_index"] = int(event_index)
    selected["real_event_start_s"] = float(event.start)
    selected["real_event_end_s"] = float(event.end)
    selected["real_event_duration_s"] = float(duration)
    position_summaries = [
        _window_position_summary(session.position, float(row.window_start_s), float(row.window_end_s))
        for row in selected.itertuples(index=False)
    ]
    if position_summaries:
        selected = pd.concat([selected.reset_index(drop=True), pd.DataFrame(position_summaries)], axis=1)
    return selected.drop(columns=["random_tiebreaker"], errors="ignore")


def _candidate_null_windows(
    session: ReplaySession,
    *,
    duration_s: float,
    candidate_step_s: float | None,
    exclusion_padding_s: float,
    restrict_to_run_times: bool,
    max_candidate_windows: int | None,
    random_seed: int,
) -> pd.DataFrame:
    base_intervals = _base_candidate_intervals(session, restrict_to_run_times=restrict_to_run_times)
    excluded = _padded_intervals(session.ripple_events[:, :2], padding_s=exclusion_padding_s)
    step = float(candidate_step_s) if candidate_step_s and candidate_step_s > 0.0 else max(duration_s / 2.0, 0.001)
    start_intervals = _valid_start_intervals(base_intervals, excluded, duration_s=float(duration_s))
    if start_intervals.size == 0:
        return pd.DataFrame()
    exhaustive_size = _exhaustive_start_count(start_intervals, step=step)
    cap = _candidate_window_cap(
        max_candidate_windows=max_candidate_windows,
        candidate_step_s=candidate_step_s,
        restrict_to_run_times=restrict_to_run_times,
    )
    if cap is not None and exhaustive_size > cap:
        starts = _sample_start_times(start_intervals, n=int(cap), random_seed=int(random_seed))
        sampling_mode = "sampled"
    else:
        starts = _exhaustive_start_times(start_intervals, step=step)
        sampling_mode = "exhaustive"
    rows: list[dict[str, float | int | bool]] = []
    for start in starts:
        end = float(start) + float(duration_s)
        rows.append(
            {
                "window_start_s": float(start),
                "window_end_s": float(end),
                "window_duration_s": float(duration_s),
                "off_swr": True,
                "restrict_to_run_times": bool(restrict_to_run_times),
                "candidate_sampling_mode": sampling_mode,
                "candidate_pool_size": int(len(starts)),
                "candidate_pool_exhaustive_size": int(exhaustive_size),
            }
        )
    return pd.DataFrame(rows)


def _candidate_window_cap(
    *,
    max_candidate_windows: int | None,
    candidate_step_s: float | None,
    restrict_to_run_times: bool,
) -> int | None:
    if candidate_step_s is not None:
        return None
    if restrict_to_run_times:
        return None
    if max_candidate_windows is None or int(max_candidate_windows) <= 0:
        return None
    return int(max_candidate_windows)


def _valid_start_intervals(base_intervals: np.ndarray, excluded: np.ndarray, *, duration_s: float) -> np.ndarray:
    intervals: list[tuple[float, float]] = []
    for interval_start, interval_end in np.asarray(base_intervals, dtype=float).reshape(-1, 2):
        start_min = float(interval_start)
        start_max = float(interval_end) - float(duration_s)
        if start_max < start_min:
            continue
        pieces = [(start_min, start_max)]
        for excluded_start, excluded_end in np.asarray(excluded, dtype=float).reshape(-1, 2):
            forbidden_start = float(excluded_start) - float(duration_s)
            forbidden_end = float(excluded_end)
            next_pieces: list[tuple[float, float]] = []
            for piece_start, piece_end in pieces:
                if forbidden_end <= piece_start or forbidden_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if forbidden_start > piece_start:
                    next_pieces.append((piece_start, min(piece_end, forbidden_start)))
                if forbidden_end < piece_end:
                    next_pieces.append((max(piece_start, forbidden_end), piece_end))
            pieces = next_pieces
            if not pieces:
                break
        intervals.extend((start, end) for start, end in pieces if end >= start)
    return np.asarray(intervals, dtype=float).reshape(-1, 2) if intervals else np.empty((0, 2), dtype=float)


def _exhaustive_start_count(start_intervals: np.ndarray, *, step: float) -> int:
    total = 0
    for start, end in np.asarray(start_intervals, dtype=float).reshape(-1, 2):
        total += int(np.floor((float(end) - float(start)) / float(step))) + 1
    return int(total)


def _exhaustive_start_times(start_intervals: np.ndarray, *, step: float) -> np.ndarray:
    starts: list[np.ndarray] = []
    for start, end in np.asarray(start_intervals, dtype=float).reshape(-1, 2):
        starts.append(np.arange(float(start), float(end) + float(step) * 0.5, float(step)))
    return np.concatenate(starts) if starts else np.array([], dtype=float)


def _sample_start_times(start_intervals: np.ndarray, *, n: int, random_seed: int) -> np.ndarray:
    intervals = np.asarray(start_intervals, dtype=float).reshape(-1, 2)
    lengths = intervals[:, 1] - intervals[:, 0]
    total = float(lengths.sum())
    if total <= 0.0 or not np.isfinite(total):
        return intervals[:, 0][: int(n)].astype(float)
    rng = np.random.default_rng(int(random_seed))
    choices = rng.choice(len(intervals), size=int(n), replace=True, p=lengths / total)
    offsets = rng.random(int(n)) * lengths[choices]
    starts = intervals[choices, 0] + offsets
    return np.sort(starts.astype(float))


def _base_candidate_intervals(session: ReplaySession, *, restrict_to_run_times: bool) -> np.ndarray:
    if restrict_to_run_times and session.run_times.size:
        return np.asarray(session.run_times, dtype=float).reshape(-1, 2)
    if session.position.size:
        position_times = np.asarray(session.position[:, 0], dtype=float)
        finite = position_times[np.isfinite(position_times)]
        if finite.size:
            return np.array([[float(np.nanmin(finite)), float(np.nanmax(finite))]], dtype=float)
    starts: list[float] = []
    ends: list[float] = []
    if session.spikes.size:
        starts.append(float(np.nanmin(session.spikes[:, 0])))
        ends.append(float(np.nanmax(session.spikes[:, 0])))
    if session.ripple_events.size:
        starts.append(float(np.nanmin(session.ripple_events[:, 0])))
        ends.append(float(np.nanmax(session.ripple_events[:, 1])))
    if not starts or not ends:
        return np.empty((0, 2), dtype=float)
    return np.array([[min(starts), max(ends)]], dtype=float)


def _padded_intervals(intervals: np.ndarray, *, padding_s: float) -> np.ndarray:
    arr = np.asarray(intervals, dtype=float).reshape(-1, 2)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    out = arr.copy()
    out[:, 0] -= float(padding_s)
    out[:, 1] += float(padding_s)
    return out


def _overlaps_any(start: float, end: float, intervals: np.ndarray) -> bool:
    if intervals.size == 0:
        return False
    return bool(np.any((start < intervals[:, 1]) & (end > intervals[:, 0])))


def _time_sorted_spikes(spikes: np.ndarray) -> np.ndarray:
    arr = np.asarray(spikes, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[0] > 1 and np.any(arr[1:, 0] < arr[:-1, 0]):
        arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
    return arr


def _spike_count_and_active_cells(spikes: np.ndarray, start: float, end: float, *, assume_sorted: bool = False) -> tuple[int, int]:
    if spikes.size == 0:
        return 0, 0
    arr = np.asarray(spikes, dtype=float)
    if not assume_sorted:
        arr = _time_sorted_spikes(arr)
    times = arr[:, 0]
    left = int(np.searchsorted(times, float(start), side="left"))
    right = int(np.searchsorted(times, float(end), side="left"))
    selected = arr[left:right]
    if selected.size == 0:
        return 0, 0
    return int(selected.shape[0]), int(np.unique(selected[:, 1].astype(int)).size)


def _position_speed_samples(position: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(position, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return np.array([], dtype=float), np.empty((0, 2), dtype=float), np.array([], dtype=float)
    finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    arr = arr[finite]
    if arr.size == 0:
        return np.array([], dtype=float), np.empty((0, 2), dtype=float), np.array([], dtype=float)
    order = np.argsort(arr[:, 0], kind="mergesort")
    arr = arr[order]
    times = arr[:, 0].astype(float)
    xy = arr[:, 1:3].astype(float)
    speed = np.full(times.shape, np.nan, dtype=float)
    if len(times) == 1:
        speed[0] = 0.0
        return times, xy, speed
    dt = np.diff(times)
    displacement = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    segment_speed = np.divide(displacement, dt, out=np.full_like(displacement, np.nan, dtype=float), where=dt > 0)
    speed[1:] = segment_speed
    speed[0] = segment_speed[0] if np.isfinite(segment_speed[0]) else np.nan
    return times, xy, speed


def _window_position_summary(position: np.ndarray, start: float, end: float) -> dict[str, object]:
    times, xy, speed = _position_speed_samples(position)
    summary = {
        "animal_speed_mean": np.nan,
        "animal_speed_median": np.nan,
        "animal_speed_max": np.nan,
        "animal_x": np.nan,
        "animal_y": np.nan,
        "position_sample_count": 0,
    }
    if times.size == 0 or not np.isfinite(start) or not np.isfinite(end):
        return summary
    mask = (times >= float(start)) & (times <= float(end))
    if not np.any(mask):
        return summary
    window_speed = speed[mask]
    window_xy = xy[mask]
    finite_speed = window_speed[np.isfinite(window_speed)]
    finite_xy = window_xy[np.isfinite(window_xy).all(axis=1)]
    summary["position_sample_count"] = int(mask.sum())
    if finite_speed.size:
        summary["animal_speed_mean"] = float(np.mean(finite_speed))
        summary["animal_speed_median"] = float(np.median(finite_speed))
        summary["animal_speed_max"] = float(np.max(finite_speed))
    if finite_xy.size:
        summary["animal_x"] = float(np.mean(finite_xy[:, 0]))
        summary["animal_y"] = float(np.mean(finite_xy[:, 1]))
    return summary


def score_matched_nulls(args: argparse.Namespace) -> pd.DataFrame:
    """Score real core windows and spike-matched null windows."""

    session_dir = _session_path(args.dataset_root, args.session)
    _check_session(session_dir)
    session = load_replay_session(session_dir)
    event_ids = _events(args.events, session)
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]

    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
            min_occupancy_s=args.min_occupancy_s,
            rate_floor_hz=args.rate_floor_hz,
        ),
    )
    models = _models(args, session, encoding=encoding)
    has_clusterless = any(isinstance(model, ClusterlessStateSpaceReplayModel) for model in models.values())
    clusterless_encoding = fit_clusterless_mark_encoding(session, _clusterless_mark_config(args)) if has_clusterless else None
    emissions_cfg = EmissionConfig(
        time_bin_s=args.time_bin_s,
        spike_rate_scale=args.spike_rate_scale,
        likelihood_temperature=args.emission_likelihood_temperature,
        negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
    )
    sorted_calibration = ReplayEmissionCalibration(
        gain_mode=args.replay_gain_mode,
        gain_prior_count=args.replay_gain_prior_count,
        max_gain=args.replay_gain_max_gain,
        emission_model=args.sorted_spike_emission_model,
        negative_binomial_dispersion=args.negative_binomial_dispersion,
    )

    rows: list[dict[str, object]] = []
    for event_id in event_ids:
        event = session.ripple(int(event_id))
        real_count, real_active = _spike_count_and_active_cells(
            session.excitatory_spikes(),
            float(event.start),
            float(event.end),
        )
        window_rows = [
            {
                "window_role": "real",
                "event_window_variant": "core",
                "null_index": -1,
                "matched_null_rank": 0,
                "template_event_index": int(event_id),
                "window_start_s": float(event.start),
                "window_end_s": float(event.end),
                "window_duration_s": float(event.end - event.start),
                "real_event_start_s": float(event.start),
                "real_event_end_s": float(event.end),
                "real_event_duration_s": float(event.end - event.start),
                "real_n_spikes": int(real_count),
                "real_active_cell_count": int(real_active),
                "null_n_spikes": int(real_count),
                "null_active_cell_count": int(real_active),
                "n_spikes_delta": 0,
                "active_cell_count_delta": 0,
                "n_spikes_relative_delta": 0.0,
                "off_swr": False,
                **_window_position_summary(session.position, float(event.start), float(event.end)),
            }
        ]
        nulls = spike_matched_null_windows(
            session,
            int(event_id),
            nulls_per_event=args.nulls_per_event,
            random_seed=args.null_random_seed,
            spike_count_tolerance_fraction=args.spike_count_tolerance_fraction,
            active_cell_tolerance=args.active_cell_tolerance,
            candidate_step_s=args.null_candidate_step_s,
            exclusion_padding_s=args.swr_exclusion_padding_s,
            restrict_to_run_times=not args.allow_non_run_nulls,
            max_candidate_windows=args.max_null_candidate_windows,
        )
        for row in nulls.to_dict("records"):
            item = dict(row)
            item["window_role"] = "matched_null"
            item["event_window_variant"] = "matched_null"
            item.update(
                _window_position_summary(
                    session.position,
                    float(item["window_start_s"]),
                    float(item["window_end_s"]),
                )
            )
            window_rows.append(item)
        for window_index, window in enumerate(window_rows):
            _score_one_window(
                args,
                session,
                encoding,
                clusterless_encoding,
                models,
                emissions_cfg,
                sorted_calibration,
                event_id=int(event_id),
                window_index=int(window_index),
                window=window,
                rows=rows,
            )
    return _postprocess_evidence_scores(pd.DataFrame(rows))


def _score_one_window(
    args: argparse.Namespace,
    session: ReplaySession,
    encoding,
    clusterless_encoding,
    models: dict[str, object],
    emissions_cfg: EmissionConfig,
    sorted_calibration: ReplayEmissionCalibration,
    *,
    event_id: int,
    window_index: int,
    window: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    event_window = SimpleNamespace(
        start=float(window["window_start_s"]),
        end=float(window["window_end_s"]),
    )
    sorted_emissions = build_sorted_emissions_with_replay_calibration(
        session,
        encoding,
        event_window,
        emissions_cfg,
        calibration=sorted_calibration,
    )
    if sorted_emissions.n_time == 0:
        return
    clusterless_emissions = (
        build_clusterless_mark_emissions(session, clusterless_encoding, event_window, emissions_cfg)
        if clusterless_encoding is not None
        else None
    )
    window_settings = _window_settings(window, window_index=window_index)
    for name, model in models.items():
        start = time.perf_counter()
        use_clusterless = isinstance(model, ClusterlessStateSpaceReplayModel)
        emissions = clusterless_emissions if use_clusterless else sorted_emissions
        bin_centers = clusterless_encoding.bin_centers if use_clusterless and clusterless_encoding is not None else encoding.bin_centers
        occupancy_s = clusterless_encoding.occupancy_s if use_clusterless and clusterless_encoding is not None else encoding.occupancy_s
        assert emissions is not None
        try:
            result = score_replay_model_compat(model, emissions, bin_centers, occupancy_s=occupancy_s)
            model_name = str(result.model_name)
            row: dict[str, object] = {
                "status": "success",
                "session": session.session_id,
                "event_index": int(event_id),
                **window_settings,
                "model": model_name,
                "requested_model": name,
                "model_family": _family(model_name),
                "log_evidence": float(result.log_likelihood),
                "n_time": int(result.n_time),
                "n_spikes": int(result.n_spikes),
                "runtime_s": float(time.perf_counter() - start),
                "error": "",
                **_run_settings(args),
                "matched_nulls_per_event": int(args.nulls_per_event),
                "spike_count_tolerance_fraction": float(args.spike_count_tolerance_fraction),
                "active_cell_tolerance": "" if args.active_cell_tolerance is None else int(args.active_cell_tolerance),
                "null_candidate_step_s": "" if args.null_candidate_step_s is None else float(args.null_candidate_step_s),
                "max_null_candidate_windows": "" if args.max_null_candidate_windows is None else int(args.max_null_candidate_windows),
                "swr_exclusion_padding_s": float(args.swr_exclusion_padding_s),
                "allow_non_run_nulls": bool(args.allow_non_run_nulls),
            }
            metadata = getattr(emissions, "metadata", {}) or {}
            row.update({f"emission_{key}": value for key, value in metadata.items()})
            row.update({f"diagnostic_{key}": value for key, value in result.diagnostics.items()})
            rows.append(row)
            print(
                f"Scored {session.session_id} event {event_id} {window_settings['window_role']} "
                f"{window_settings['null_index']} with {name}",
                flush=True,
            )
        except Exception as exc:
            rows.append(
                {
                    "status": "failure",
                    "session": session.session_id,
                    "event_index": int(event_id),
                    **window_settings,
                    "model": name,
                    "requested_model": name,
                    "model_family": _family(name),
                    "log_evidence": np.nan,
                    "n_time": int(emissions.n_time),
                    "n_spikes": int(emissions.n_spikes),
                    "runtime_s": float(time.perf_counter() - start),
                    "error": f"{type(exc).__name__}: {exc}",
                    **_run_settings(args),
                }
            )
            if not args.continue_on_error:
                raise


def _window_settings(window: dict[str, object], *, window_index: int) -> dict[str, object]:
    keys = [
        "window_role",
        "event_window_variant",
        "null_index",
        "matched_null_rank",
        "template_event_index",
        "window_start_s",
        "window_end_s",
        "window_duration_s",
        "real_event_start_s",
        "real_event_end_s",
        "real_event_duration_s",
        "real_n_spikes",
        "real_active_cell_count",
        "null_n_spikes",
        "null_active_cell_count",
        "n_spikes_delta",
        "active_cell_count_delta",
        "n_spikes_relative_delta",
        "off_swr",
        "restrict_to_run_times",
        "candidate_sampling_mode",
        "candidate_pool_size",
        "candidate_pool_exhaustive_size",
        "animal_speed_mean",
        "animal_speed_median",
        "animal_speed_max",
        "animal_x",
        "animal_y",
        "position_sample_count",
    ]
    out = {key: window.get(key, "") for key in keys}
    out["window_index"] = int(window_index)
    return out


def _clusterless_mark_config(args: argparse.Namespace):
    from benchmark_model_evidence_improved import _clusterless_mark_config as build_config

    return build_config(args)


def matched_null_family_margin_decisions(
    frame: pd.DataFrame,
    *,
    comparison_scope: str = "auto",
    required_models: tuple[str, ...] | None = None,
    trajectory_models: tuple[str, ...] | None = None,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    """Return best exact trajectory versus nontrajectory decisions for real/null windows."""

    if frame.empty:
        return pd.DataFrame()
    scored_models = _scored_model_names(frame)
    resolved_required, resolved_trajectory = resolve_family_model_sets(
        comparison_scope=comparison_scope,
        scored_models=scored_models,
    )
    required = tuple(str(model) for model in (required_models or resolved_required))
    required_set = set(required)
    trajectory = tuple(str(model) for model in (trajectory_models or resolved_trajectory))
    trajectory_set = set(trajectory)
    scope_label = _canonical_comparison_scope(comparison_scope, required)
    status_ok = frame["status"].eq("success") if "status" in frame else pd.Series(True, index=frame.index)
    comparable = _coerce_bool_series(frame["evidence_comparable"], default=False) if "evidence_comparable" in frame else pd.Series(True, index=frame.index)
    ok = frame[status_ok & comparable].copy()
    rows: list[dict[str, object]] = []
    group_cols = ["session", "event_index", "window_role", "null_index"]
    for key, group in ok.groupby(group_cols, sort=True):
        session, event_index, window_role, null_index = key
        core = group[group["model"].astype(str).isin(required_set)].dropna(subset=["log_evidence"]).copy()
        present = tuple(model for model in required if model in set(core["model"].astype(str)))
        missing = tuple(model for model in required if model not in set(present))
        trajectory = core[core["model"].astype(str).isin(trajectory_set)]
        nontrajectory = core[~core["model"].astype(str).isin(trajectory_set)]
        if trajectory.empty or nontrajectory.empty:
            best_trajectory_model = ""
            best_trajectory_log_evidence = np.nan
            best_nontrajectory_model = ""
            best_nontrajectory_log_evidence = np.nan
            margin = np.nan
            decision = "incomplete_core"
        else:
            best_trajectory = trajectory.sort_values("log_evidence", ascending=False).iloc[0]
            best_nontrajectory = nontrajectory.sort_values("log_evidence", ascending=False).iloc[0]
            best_trajectory_model = str(best_trajectory["model"])
            best_trajectory_log_evidence = float(best_trajectory["log_evidence"])
            best_nontrajectory_model = str(best_nontrajectory["model"])
            best_nontrajectory_log_evidence = float(best_nontrajectory["log_evidence"])
            margin = best_trajectory_log_evidence - best_nontrajectory_log_evidence
            if missing:
                decision = "incomplete_core"
            elif margin >= float(margin_threshold):
                decision = "trajectory"
            elif margin <= -float(margin_threshold):
                decision = "nontrajectory"
            else:
                decision = "ambiguous"
        n_spikes = _first_numeric_value(group, "n_spikes")
        n_time = _first_numeric_value(group, "n_time")
        rows.append(
            {
                "session": str(session),
                "rat": str(session).split("/")[0],
                "event_index": int(event_index),
                "window_role": str(window_role),
                "null_index": int(null_index),
                "window_start_s": _first_numeric_value(group, "window_start_s"),
                "window_end_s": _first_numeric_value(group, "window_end_s"),
                "window_duration_s": _first_numeric_value(group, "window_duration_s"),
                "n_spikes": n_spikes,
                "n_time": n_time,
                "active_cell_count": _first_numeric_value(group, "null_active_cell_count"),
                "real_n_spikes": _first_numeric_value(group, "real_n_spikes"),
                "n_spikes_delta": _first_numeric_value(group, "n_spikes_delta"),
                "n_spikes_relative_delta": _first_numeric_value(group, "n_spikes_relative_delta"),
                "comparison_scope": scope_label,
                "required_models_present": int(len(present)),
                "required_models_total": int(len(required)),
                "required_models_complete": bool(not missing),
                "missing_required_models": " ".join(missing),
                "margin_threshold": float(margin_threshold),
                "best_trajectory_model": best_trajectory_model,
                "best_trajectory_log_evidence": best_trajectory_log_evidence,
                "best_nontrajectory_model": best_nontrajectory_model,
                "best_nontrajectory_log_evidence": best_nontrajectory_log_evidence,
                "trajectory_minus_nontrajectory_log_evidence": margin,
                "best_trajectory_log_evidence_per_time_bin": _safe_ratio(best_trajectory_log_evidence, n_time),
                "best_trajectory_log_evidence_per_spike": _safe_ratio(best_trajectory_log_evidence, n_spikes),
                "trajectory_minus_nontrajectory_log_evidence_per_time_bin": _safe_ratio(margin, n_time),
                "trajectory_minus_nontrajectory_log_evidence_per_spike": _safe_ratio(margin, n_spikes),
                "trajectory_confident_claim": bool(decision == "trajectory"),
                "nontrajectory_confident_claim": bool(decision == "nontrajectory"),
                "margin_decision": decision,
            }
        )
    return pd.DataFrame(rows)


def matched_null_family_margin_summary(
    decisions: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("comparison_scope", "window_role"),
) -> pd.DataFrame:
    """Summarize real/null family-margin decisions."""

    if decisions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, group in decisions.groupby(list(group_cols), sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        margins = pd.to_numeric(group["trajectory_minus_nontrajectory_log_evidence"], errors="coerce")
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(
            {
                "windows": int(len(group)),
                "events": int(group[["session", "event_index"]].drop_duplicates().shape[0]),
                "required_complete_windows": int(_coerce_bool_series(group["required_models_complete"]).sum()),
                "trajectory_confident_claims": int(_coerce_bool_series(group["trajectory_confident_claim"]).sum()),
                "nontrajectory_confident_claims": int(_coerce_bool_series(group["nontrajectory_confident_claim"]).sum()),
                "ambiguous_windows": int((group["margin_decision"] == "ambiguous").sum()),
                "mean_family_margin": float(margins.mean()),
                "median_family_margin": float(margins.median()),
                "mean_best_trajectory_log_evidence_per_spike": float(
                    pd.to_numeric(group["best_trajectory_log_evidence_per_spike"], errors="coerce").mean()
                ),
                "median_best_trajectory_log_evidence_per_spike": float(
                    pd.to_numeric(group["best_trajectory_log_evidence_per_spike"], errors="coerce").median()
                ),
                "mean_best_trajectory_log_evidence_per_time_bin": float(
                    pd.to_numeric(group["best_trajectory_log_evidence_per_time_bin"], errors="coerce").mean()
                ),
                "median_best_trajectory_log_evidence_per_time_bin": float(
                    pd.to_numeric(group["best_trajectory_log_evidence_per_time_bin"], errors="coerce").median()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def matched_null_empirical_p_values(decisions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-event empirical p-values from matched-null margins."""

    if decisions.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (session, event_index), group in decisions.groupby(["session", "event_index"], sort=True):
        real = group[group["window_role"].astype(str).eq("real")]
        nulls = group[group["window_role"].astype(str).eq("matched_null")]
        if real.empty:
            continue
        real_row = real.sort_values("null_index").iloc[0]
        null_margin = pd.to_numeric(nulls["trajectory_minus_nontrajectory_log_evidence"], errors="coerce").dropna()
        real_margin = float(real_row["trajectory_minus_nontrajectory_log_evidence"])
        null_best_per_spike = pd.to_numeric(nulls["best_trajectory_log_evidence_per_spike"], errors="coerce").dropna()
        null_best_per_time = pd.to_numeric(nulls["best_trajectory_log_evidence_per_time_bin"], errors="coerce").dropna()
        k = int(len(null_margin))
        p_value = empirical_p_value(real_margin, null_margin)
        rows.append(
            {
                "comparison_scope": str(real_row.get("comparison_scope", "")),
                "session": str(session),
                "rat": str(session).split("/")[0],
                "event_index": int(event_index),
                "matched_null_windows": k,
                "minimum_possible_empirical_p_value": float(1.0 / (1.0 + k)) if k else np.nan,
                "empirical_p_value": p_value,
                "real_family_margin": real_margin,
                "median_null_family_margin": float(null_margin.median()) if k else np.nan,
                "mean_null_family_margin": float(null_margin.mean()) if k else np.nan,
                "real_minus_median_null_family_margin": real_margin - float(null_margin.median()) if k else np.nan,
                "real_best_trajectory_log_evidence_per_spike": float(real_row["best_trajectory_log_evidence_per_spike"]),
                "median_null_best_trajectory_log_evidence_per_spike": float(null_best_per_spike.median()) if not null_best_per_spike.empty else np.nan,
                "real_minus_median_null_best_trajectory_log_evidence_per_spike": (
                    float(real_row["best_trajectory_log_evidence_per_spike"]) - float(null_best_per_spike.median())
                    if not null_best_per_spike.empty
                    else np.nan
                ),
                "real_best_trajectory_log_evidence_per_time_bin": float(real_row["best_trajectory_log_evidence_per_time_bin"]),
                "median_null_best_trajectory_log_evidence_per_time_bin": float(null_best_per_time.median()) if not null_best_per_time.empty else np.nan,
                "real_minus_median_null_best_trajectory_log_evidence_per_time_bin": (
                    float(real_row["best_trajectory_log_evidence_per_time_bin"]) - float(null_best_per_time.median())
                    if not null_best_per_time.empty
                    else np.nan
                ),
                "real_trajectory_confident_claim": bool(real_row["trajectory_confident_claim"]),
                "real_nontrajectory_confident_claim": bool(real_row["nontrajectory_confident_claim"]),
            }
        )
    return pd.DataFrame(rows)


def empirical_p_value(
    real_value: float,
    null_values: tuple[float, ...] | list[float] | np.ndarray | pd.Series,
    *,
    greater_equal: bool = True,
) -> float:
    """Monte Carlo empirical p-value with +1 correction.

    For K null samples, the minimum possible p-value is 1 / (K + 1).
    """

    null = np.asarray(null_values, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(real_value):
        return float("nan")
    if greater_equal:
        exceed = int(np.sum(null >= float(real_value)))
    else:
        exceed = int(np.sum(null <= float(real_value)))
    return float((1 + exceed) / (1 + null.size))


def matched_null_group_summary(p_values: pd.DataFrame, *, group_cols: tuple[str, ...]) -> pd.DataFrame:
    if p_values.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, group in p_values.groupby(list(group_cols), sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        delta = pd.to_numeric(group["real_minus_median_null_family_margin"], errors="coerce")
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        row.update(_delta_summary(group, delta))
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_rat_out_matched_null_summary(p_values: pd.DataFrame) -> pd.DataFrame:
    if p_values.empty or "rat" not in p_values:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    scope_values = sorted(p_values["comparison_scope"].dropna().astype(str).unique()) if "comparison_scope" in p_values else [""]
    for scope in scope_values:
        scoped = p_values[p_values["comparison_scope"].astype(str).eq(scope)] if scope else p_values
        for rat in sorted(scoped["rat"].dropna().astype(str).unique()):
            group = scoped[scoped["rat"].astype(str) != rat]
            delta = pd.to_numeric(group["real_minus_median_null_family_margin"], errors="coerce")
            row = {"comparison_scope": scope, "held_out_rat": rat}
            row.update(_delta_summary(group, delta))
            rows.append(row)
    return pd.DataFrame(rows)


def rat_bootstrap_matched_null_summary(
    p_values: pd.DataFrame,
    *,
    random_seed: int = 1,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    if p_values.empty or "rat" not in p_values:
        return pd.DataFrame()
    scope_values = sorted(p_values["comparison_scope"].dropna().astype(str).unique()) if "comparison_scope" in p_values else [""]
    if not scope_values:
        scope_values = [""]
    rows: list[dict[str, object]] = []
    for scope_index, scope in enumerate(scope_values):
        scoped = p_values[p_values["comparison_scope"].astype(str).eq(scope)] if scope else p_values
        rats = sorted(scoped["rat"].dropna().astype(str).unique())
        if not rats:
            continue
        rng = np.random.default_rng(int(random_seed) + 1009 * int(scope_index))
        mean_values: list[float] = []
        median_values: list[float] = []
        for _ in range(int(n_bootstrap)):
            sampled = rng.choice(rats, size=len(rats), replace=True)
            pieces = [scoped[scoped["rat"].astype(str).eq(rat)] for rat in sampled]
            sample = pd.concat(pieces, ignore_index=True)
            delta = pd.to_numeric(sample["real_minus_median_null_family_margin"], errors="coerce").dropna()
            if delta.empty:
                continue
            mean_values.append(float(delta.mean()))
            median_values.append(float(delta.median()))
        rows.append(
            {
                "comparison_scope": scope,
                "bootstrap_unit": "rat",
                "bootstrap_samples": int(len(mean_values)),
                "mean_delta_lower_95": float(np.quantile(mean_values, 0.025)) if mean_values else np.nan,
                "mean_delta_median": float(np.quantile(mean_values, 0.5)) if mean_values else np.nan,
                "mean_delta_upper_95": float(np.quantile(mean_values, 0.975)) if mean_values else np.nan,
                "median_delta_lower_95": float(np.quantile(median_values, 0.025)) if median_values else np.nan,
                "median_delta_median": float(np.quantile(median_values, 0.5)) if median_values else np.nan,
                "median_delta_upper_95": float(np.quantile(median_values, 0.975)) if median_values else np.nan,
            }
        )
    return pd.DataFrame(rows)


def matched_null_control_gate_summary(p_values: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if p_values.empty:
        return pd.DataFrame(
            [{"gate": "overall", "passed": False, "observed": "no events", "criterion": "matched-null events exist"}]
        )
    delta = pd.to_numeric(p_values["real_minus_median_null_family_margin"], errors="coerce")
    _append_gate(rows, "median_real_minus_null_family_margin_positive", float(delta.median()) > 0.0, float(delta.median()), "median real-minus-null family margin > 0")
    _append_gate(rows, "majority_events_exceed_null_median", float((delta > 0.0).mean()) > 0.5, float((delta > 0.0).mean()), "majority of events have real margin > matched-null median")
    rat_summary = matched_null_group_summary(p_values, group_cols=("rat",))
    rat_medians = pd.to_numeric(rat_summary["median_real_minus_median_null_family_margin"], errors="coerce") if not rat_summary.empty else pd.Series(dtype=float)
    _append_gate(rows, "per_rat_median_positive", bool(not rat_medians.empty and (rat_medians > 0.0).all()), "" if rat_medians.empty else float(rat_medians.min()), "per-rat median real-minus-null margin > 0")
    loo = leave_one_rat_out_matched_null_summary(p_values)
    loo_medians = pd.to_numeric(loo["median_real_minus_median_null_family_margin"], errors="coerce") if not loo.empty else pd.Series(dtype=float)
    _append_gate(rows, "leave_one_rat_out_median_positive", bool(not loo_medians.empty and (loo_medians > 0.0).all()), "" if loo_medians.empty else float(loo_medians.min()), "leave-one-rat-out median real-minus-null margin > 0")
    if not bootstrap.empty:
        mean_lower = float(bootstrap["mean_delta_lower_95"].iloc[0])
        median_lower = float(bootstrap["median_delta_lower_95"].iloc[0])
        _append_gate(rows, "rat_bootstrap_lower_bound_positive", bool(mean_lower > 0.0 or median_lower > 0.0), f"mean={mean_lower:.3f}; median={median_lower:.3f}", "rat-bootstrap lower bound for mean or median real-minus-null margin > 0")
    nontrajectory = int(_coerce_bool_series(p_values["real_nontrajectory_confident_claim"]).sum())
    _append_gate(rows, "real_nontrajectory_claims_near_zero", nontrajectory <= max(1, math.ceil(0.05 * len(p_values))), nontrajectory, "false/nontrajectory claims remain near zero")
    rows.append(
        {
            "gate": "overall",
            "passed": all(bool(row["passed"]) for row in rows),
            "observed": f"{sum(bool(row['passed']) for row in rows)}/{len(rows)} gates passed",
            "criterion": "all matched-null control gates pass",
        }
    )
    return pd.DataFrame(rows)


def targeted_matched_null_event_diagnostics(p_values: pd.DataFrame) -> pd.DataFrame:
    """Return per-event high-K matched-null diagnostics."""

    columns = [
        "comparison_scope",
        "session",
        "rat",
        "event_index",
        "real_family_margin",
        "median_null_family_margin",
        "mean_null_family_margin",
        "real_minus_median_null_family_margin",
        "empirical_p_value",
        "matched_null_windows",
        "minimum_possible_empirical_p_value",
        "real_best_trajectory_log_evidence_per_spike",
        "median_null_best_trajectory_log_evidence_per_spike",
        "real_minus_median_null_best_trajectory_log_evidence_per_spike",
        "real_best_trajectory_log_evidence_per_time_bin",
        "median_null_best_trajectory_log_evidence_per_time_bin",
        "real_minus_median_null_best_trajectory_log_evidence_per_time_bin",
    ]
    if p_values.empty:
        return pd.DataFrame(columns=columns)
    return p_values[[column for column in columns if column in p_values]].copy()


def targeted_matched_null_session_diagnostics(p_values: pd.DataFrame) -> pd.DataFrame:
    """Return session-level high-K matched-null diagnostics."""

    if p_values.empty:
        return pd.DataFrame()
    group_cols = tuple(column for column in ("comparison_scope", "session") if column in p_values)
    return matched_null_group_summary(p_values, group_cols=group_cols)


def lightweight_matched_null_control_gate_summary(
    p_values: pd.DataFrame,
    decisions: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """Return gates for lightweight first-order-IMM-vs-stationary diagnostics."""

    rows: list[dict[str, object]] = []
    if p_values.empty:
        return pd.DataFrame(
            [{"gate": "overall", "passed": False, "observed": "no events", "criterion": "matched-null events exist"}]
        )
    scope_values = set(p_values.get("comparison_scope", pd.Series(dtype=str)).dropna().astype(str))
    lightweight = scope_values == {LIGHTWEIGHT_FO_IMM_STATIONARY_SCOPE}
    _append_gate(
        rows,
        "comparison_scope_lightweight",
        lightweight,
        " ".join(sorted(scope_values)) if scope_values else "",
        f"comparison_scope == {LIGHTWEIGHT_FO_IMM_STATIONARY_SCOPE}",
    )
    complete = _coerce_bool_series(decisions["required_models_complete"]) if not decisions.empty else pd.Series(dtype=bool)
    _append_gate(
        rows,
        "complete_lightweight_family_rows",
        bool(lightweight and not complete.empty and complete.all()),
        "" if complete.empty else f"{int(complete.sum())}/{len(complete)}",
        "all lightweight family rows have required two-model evidence",
    )
    delta = pd.to_numeric(p_values["real_minus_median_null_family_margin"], errors="coerce")
    sessions = sorted(p_values["session"].dropna().astype(str).unique())
    targeted_session = lightweight and len(sessions) == 1
    if targeted_session:
        _append_gate(rows, "session_median_positive", float(delta.median()) > 0.0, float(delta.median()), "target session median real-minus-null margin > 0")
        _append_gate(rows, "session_majority_events_exceed_null_median", float((delta > 0.0).mean()) > 0.5, float((delta > 0.0).mean()), "target session majority events exceed null median")
        p_values_numeric = pd.to_numeric(p_values["empirical_p_value"], errors="coerce")
        low_p_events = int((p_values_numeric <= 0.10).sum())
        _append_gate(rows, "session_has_at_least_one_p_le_0_05_or_p_le_0_10", low_p_events >= 1, low_p_events, "target session has at least one event with empirical p <= 0.10")
    else:
        _append_gate(rows, "median_real_minus_null_family_margin_positive", float(delta.median()) > 0.0, float(delta.median()), "median real-minus-null family margin > 0")
        _append_gate(rows, "majority_events_exceed_null_median", float((delta > 0.0).mean()) > 0.5, float((delta > 0.0).mean()), "majority of events have real margin > matched-null median")
        session_summary = matched_null_group_summary(p_values, group_cols=("session",))
        session_medians = pd.to_numeric(session_summary["median_real_minus_median_null_family_margin"], errors="coerce") if not session_summary.empty else pd.Series(dtype=float)
        _append_gate(rows, "per_session_median_positive", bool(not session_medians.empty and (session_medians > 0.0).all()), "" if session_medians.empty else float(session_medians.min()), "per-session median real-minus-null margin > 0")
        rat_summary = matched_null_group_summary(p_values, group_cols=("rat",))
        rat_medians = pd.to_numeric(rat_summary["median_real_minus_median_null_family_margin"], errors="coerce") if not rat_summary.empty else pd.Series(dtype=float)
        _append_gate(rows, "per_rat_median_positive", bool(not rat_medians.empty and (rat_medians > 0.0).all()), "" if rat_medians.empty else float(rat_medians.min()), "per-rat median real-minus-null margin > 0")
        loo = leave_one_rat_out_matched_null_summary(p_values)
        loo_medians = pd.to_numeric(loo["median_real_minus_median_null_family_margin"], errors="coerce") if not loo.empty else pd.Series(dtype=float)
        _append_gate(rows, "leave_one_rat_out_median_positive", bool(not loo_medians.empty and (loo_medians > 0.0).all()), "" if loo_medians.empty else float(loo_medians.min()), "leave-one-rat-out median real-minus-null margin > 0")
        if not bootstrap.empty:
            mean_lower = float(bootstrap["mean_delta_lower_95"].iloc[0])
            median_lower = float(bootstrap["median_delta_lower_95"].iloc[0])
            _append_gate(rows, "rat_bootstrap_lower_bound_positive", bool(mean_lower > 0.0 or median_lower > 0.0), f"mean={mean_lower:.3f}; median={median_lower:.3f}", "rat-bootstrap lower bound for mean or median real-minus-null margin > 0")
    min_possible_p = pd.to_numeric(p_values["minimum_possible_empirical_p_value"], errors="coerce")
    _append_gate(
        rows,
        "conventional_p_value_resolution_available",
        bool(not min_possible_p.dropna().empty and min_possible_p.min() <= 0.05),
        "" if min_possible_p.dropna().empty else float(min_possible_p.min()),
        "minimum possible empirical p-value <= 0.05",
    )
    nontrajectory = int(_coerce_bool_series(p_values["real_nontrajectory_confident_claim"]).sum())
    _append_gate(rows, "real_nontrajectory_claims_near_zero", nontrajectory <= max(1, math.ceil(0.05 * len(p_values))), nontrajectory, "false/nontrajectory claims remain near zero")
    rows.append(
        {
            "gate": "overall",
            "passed": all(bool(row["passed"]) for row in rows),
            "observed": f"{sum(bool(row['passed']) for row in rows)}/{len(rows)} gates passed",
            "criterion": "all lightweight matched-null gates pass",
        }
    )
    return pd.DataFrame(rows)


def _delta_summary(group: pd.DataFrame, delta: pd.Series) -> dict[str, object]:
    delta = delta.dropna()
    p_values = pd.to_numeric(group["empirical_p_value"], errors="coerce")
    null_windows = pd.to_numeric(group["matched_null_windows"], errors="coerce")
    return {
        "events": int(group[["session", "event_index"]].drop_duplicates().shape[0]),
        "matched_null_windows_per_event": float(null_windows.median()) if not null_windows.dropna().empty else np.nan,
        "median_real_minus_median_null_family_margin": float(delta.median()) if not delta.empty else np.nan,
        "mean_real_minus_median_null_family_margin": float(delta.mean()) if not delta.empty else np.nan,
        "fraction_real_margin_above_null_median": float((delta > 0.0).mean()) if not delta.empty else np.nan,
        "median_empirical_p_value": float(p_values.median()),
        "events_empirical_p_le_0_05": int((p_values <= 0.05).sum()),
        "events_empirical_p_le_0_10": int((p_values <= 0.10).sum()),
        "min_empirical_p_value": float(p_values.min()) if not p_values.dropna().empty else np.nan,
        "real_trajectory_confident_claims": int(_coerce_bool_series(group["real_trajectory_confident_claim"]).sum()),
        "real_nontrajectory_confident_claims": int(_coerce_bool_series(group["real_nontrajectory_confident_claim"]).sum()),
    }


def _append_gate(rows: list[dict[str, object]], gate: str, passed: bool, observed: object, criterion: str) -> None:
    rows.append({"gate": gate, "passed": bool(passed), "observed": observed, "criterion": criterion})


def _scored_model_names(frame: pd.DataFrame) -> tuple[str, ...]:
    if "model" not in frame:
        return ()
    return tuple(sorted(set(frame["model"].dropna().astype(str))))


def _canonical_comparison_scope(comparison_scope: str, required_models: tuple[str, ...]) -> str:
    scope = str(comparison_scope).strip().lower()
    if scope in {"auto", "from-models"}:
        if tuple(required_models) == LIGHTWEIGHT_FO_IMM_STATIONARY_REQUIRED_MODELS:
            return LIGHTWEIGHT_FO_IMM_STATIONARY_SCOPE
        if tuple(required_models) == FULL_CORE_REQUIRED_MODELS:
            return "full-core"
    if scope == "full_core":
        return "full-core"
    if scope in {"lightweight_fo_imm_stationary", "fo-imm-vs-stationary"}:
        return LIGHTWEIGHT_FO_IMM_STATIONARY_SCOPE
    return scope


def _first_numeric_value(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _safe_ratio(value: float, denominator: float) -> float:
    if not np.isfinite(value) or not np.isfinite(denominator) or float(denominator) == 0.0:
        return np.nan
    return float(value) / float(denominator)


def aggregate_matched_null_scores(
    score_glob: str,
    outdir: Path,
    *,
    comparison_scope: str = "auto",
    required_models: tuple[str, ...] | None = None,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    bootstrap_seed: int = 1,
    bootstrap_samples: int = 2000,
) -> pd.DataFrame:
    paths = [Path(path) for path in sorted(glob.glob(str(score_glob), recursive=True))]
    if not paths:
        raise FileNotFoundError(f"no matched-null score files found for {score_glob!r}")
    scores = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(outdir / "matched_null_event_model_evidence.csv", index=False)
    decisions = matched_null_family_margin_decisions(
        scores,
        comparison_scope=comparison_scope,
        required_models=required_models,
        margin_threshold=margin_threshold,
    )
    decisions.to_csv(outdir / "matched_null_family_margin_decisions.csv", index=False)
    matched_null_family_margin_summary(decisions).to_csv(outdir / "matched_null_family_margin_summary.csv", index=False)
    p_values = matched_null_empirical_p_values(decisions)
    p_values.to_csv(outdir / "matched_null_empirical_p_values.csv", index=False)
    matched_null_group_summary(p_values, group_cols=("comparison_scope", "session")).to_csv(outdir / "session_matched_null_summary.csv", index=False)
    matched_null_group_summary(p_values, group_cols=("comparison_scope", "rat")).to_csv(outdir / "rat_matched_null_summary.csv", index=False)
    leave_one_rat_out_matched_null_summary(p_values).to_csv(outdir / "leave_one_rat_out_matched_null_summary.csv", index=False)
    bootstrap = rat_bootstrap_matched_null_summary(p_values, random_seed=bootstrap_seed, n_bootstrap=bootstrap_samples)
    bootstrap.to_csv(outdir / "rat_bootstrap_matched_null_summary.csv", index=False)
    matched_null_control_gate_summary(p_values, bootstrap).to_csv(outdir / "matched_null_control_gate_summary.csv", index=False)
    targeted_matched_null_session_diagnostics(p_values).to_csv(outdir / "targeted_matched_null_session_diagnostics.csv", index=False)
    targeted_matched_null_event_diagnostics(p_values).to_csv(outdir / "targeted_matched_null_event_diagnostics.csv", index=False)
    lightweight_matched_null_control_gate_summary(p_values, decisions, bootstrap).to_csv(
        outdir / "lightweight_matched_null_control_gate_summary.csv",
        index=False,
    )
    return scores


def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="run:0-10")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--models", default=DEFAULT_MATCHED_NULL_MODELS)
    parser.add_argument("--nulls-per-event", type=int, default=10)
    parser.add_argument("--null-random-seed", type=int, default=1)
    parser.add_argument("--spike-count-tolerance-fraction", type=float, default=0.10)
    parser.add_argument("--active-cell-tolerance", type=int)
    parser.add_argument("--null-candidate-step-s", type=float)
    parser.add_argument("--max-null-candidate-windows", type=int, default=DEFAULT_MAX_NON_RUN_CANDIDATE_WINDOWS)
    parser.add_argument("--swr-exclusion-padding-s", type=float, default=0.0)
    parser.add_argument("--allow-non-run-nulls", action="store_true")
    _add_model_arguments(parser)
    parser.add_argument("--output", default="results/spike-matched-event-window-null")
    parser.add_argument(
        "--required-models",
        default="",
        help=(
            "Optional whitespace-separated model list required for complete "
            "family-margin decisions. Defaults to the paper full-core exact set."
        ),
    )
    _add_comparison_scope_argument(parser)
    parser.add_argument("--continue-on-error", action="store_true")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument("--stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--diffusion-sigma-cm", type=float, default=12.0)
    parser.add_argument("--momentum-sigma-cm", type=float, default=12.0)
    parser.add_argument("--velocity-decay", type=float, default=0.95)
    parser.add_argument("--mode-stickiness", type=float, default=0.94)
    parser.add_argument("--state-space-stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=60.0)
    parser.add_argument("--state-space-max-step-sigma", type=float, default=3.0)
    parser.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument("--state-space-imm-switch-tau-s", type=float, default=DEFAULT_IMPROVED_STATE_SPACE_IMM_SWITCH_TAU_S)
    parser.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=50.0)
    parser.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=45.0)
    parser.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.93)
    parser.add_argument("--state-space-momentum-velocity-decay-tau-s", type=float, default=0.0)
    parser.add_argument("--state-space-momentum-candidate-top-k", type=int, default=128)
    parser.add_argument("--state-space-momentum-candidate-mass-threshold", type=float)
    parser.add_argument("--state-space-momentum-candidate-min-k", type=int, default=1)
    parser.add_argument("--state-space-momentum-candidate-max-k", type=int, default=0)
    parser.add_argument("--state-space-momentum-predicted-candidate-top-k", type=int, default=DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_PREDICTED_CANDIDATE_TOP_K)
    parser.add_argument("--state-space-momentum-candidate-source", choices=("emission", "posterior"), default="emission")
    parser.add_argument("--state-space-valid-occupancy-threshold-s", type=float, default=0.0)
    parser.add_argument("--goal-state-space-transition-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--goal-state-space-drift-speed-cm-s", type=float, default=400.0)
    parser.add_argument("--goal-state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--clusterless-mark-likelihood", default="local-kde")
    parser.add_argument("--clusterless-mark-group-by", choices=("auto", "none", "tetrode", "cell"), default="auto")
    parser.add_argument("--clusterless-mark-smoothing-sigma-bins", type=float, default=1.0)
    parser.add_argument("--clusterless-mark-prior-count", type=float, default=1.0)
    parser.add_argument("--clusterless-mark-variance-floor", type=float, default=1.0)
    parser.add_argument("--clusterless-rate-floor-hz", type=float, default=1e-4)
    parser.add_argument("--clusterless-mark-kde-bandwidth", type=float)
    parser.add_argument("--clusterless-mark-kde-spatial-sigma-bins", type=float)
    parser.add_argument("--clusterless-mark-kde-max-neighbors", type=int, default=256)
    parser.add_argument("--time-bin-s", type=float, default=0.004)
    parser.add_argument("--spike-rate-scale", type=float, default=2.0)
    parser.add_argument("--emission-likelihood-temperature", type=float, default=0.300)
    parser.add_argument("--emission-negative-binomial-overdispersion", type=float, default=0.0)
    parser.add_argument("--sorted-spike-emission-model", choices=("poisson", "negative-binomial", "gamma-poisson"), default="poisson")
    parser.add_argument("--replay-gain-mode", choices=("none", "event", "cell", "event-cell"), default="none")
    parser.add_argument("--replay-gain-prior-count", type=float, default=10.0)
    parser.add_argument("--replay-gain-max-gain", type=float, default=20.0)
    parser.add_argument("--negative-binomial-dispersion", type=float, default=50.0)
    parser.add_argument("--include-clusterless-defaults", action="store_true")
    parser.add_argument("--valid-state-min-occupancy-s", type=float, default=0.02)
    parser.add_argument("--valid-state-top-occupancy-fraction", type=float, default=None)
    parser.add_argument("--valid-state-sigma-cm", type=float, default=5.0)
    parser.add_argument("--valid-state-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--valid-state-grid-diagonal-neighbors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--valid-state-grid-stay-probability", type=float, default=0.0)
    parser.add_argument("--window-variant-specs", default="")
    parser.add_argument("--window-pre-pads-s", default="0.0")
    parser.add_argument("--window-post-pads-s", default="0.0")
    parser.add_argument("--window-min-duration-s", type=float, default=0.004)
    parser.add_argument("--reliability-min-spikes", type=int, default=5)
    parser.add_argument("--reliability-min-time-bins", type=int, default=2)
    parser.add_argument("--reliability-max-terminal-entropy", type=float, default=float("nan"))
    parser.add_argument("--reliability-min-candidate-log-mass", type=float, default=-0.01)
    parser.add_argument("--null-shuffles", type=int, default=0)
    parser.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    parser.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    parser.add_argument("--min-occupancy-s", type=float, default=EncodingConfig().min_occupancy_s)
    parser.add_argument("--rate-floor-hz", type=float, default=EncodingConfig().rate_floor_hz)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score")
    _add_scoring_arguments(score_parser)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--score-glob", required=True)
    aggregate_parser.add_argument("--output", default="results/spike-matched-event-window-null")
    aggregate_parser.add_argument(
        "--required-models",
        default="",
        help=(
            "Optional whitespace-separated model list required for complete "
            "family-margin decisions. Defaults to the paper full-core exact set."
        ),
    )
    _add_comparison_scope_argument(aggregate_parser)
    aggregate_parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)
    aggregate_parser.add_argument("--bootstrap-seed", type=int, default=1)
    aggregate_parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    if args.command == "score":
        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        scores = score_matched_nulls(args)
        if scores.empty:
            raise RuntimeError("No matched-null scores were generated.")
        scores.to_csv(outdir / "matched_null_event_model_evidence.csv", index=False)
        aggregate_matched_null_scores(
            str(outdir / "matched_null_event_model_evidence.csv"),
            outdir,
            comparison_scope=args.comparison_scope,
            required_models=_parse_required_models(args.required_models),
            bootstrap_seed=args.null_random_seed,
        )
        return 0
    aggregate_matched_null_scores(
        args.score_glob,
        Path(args.output),
        comparison_scope=args.comparison_scope,
        required_models=_parse_required_models(args.required_models),
        margin_threshold=args.margin_threshold,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    return 0


def _add_comparison_scope_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--comparison-scope",
        default="auto",
        choices=[
            "auto",
            "full-core",
            LIGHTWEIGHT_FO_IMM_STATIONARY_SCOPE,
        ],
        help=(
            "Which required model set to use for family-margin completeness and gates. "
            "Use lightweight-first-order-imm-vs-stationary for K=50/K=100 lightweight null diagnostics."
        ),
    )


def _parse_required_models(value: str) -> tuple[str, ...] | None:
    models = tuple(item for item in str(value).split() if item)
    return models or None


if __name__ == "__main__":
    raise SystemExit(main())
