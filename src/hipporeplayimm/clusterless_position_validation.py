"""Clusterless behavioral position-decoding validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .clusterless import ClusterlessMarkConfig, build_clusterless_mark_emissions
from .clusterless_cv_exclusion import fit_clusterless_mark_encoding_excluding_intervals
from .data import ReplaySession, load_open_field_sessions
from .encoding import EmissionConfig, _clean_position, _speed_cm_s, _times_in_intervals
from .position_validation import _decode_windows, _distance


@dataclass(frozen=True)
class ClusterlessPositionValidationConfig:
    clusterless: ClusterlessMarkConfig = ClusterlessMarkConfig()
    decode_bin_s: float = 1.0
    n_folds: int = 5
    max_windows_per_session: int | None = None
    random_seed: int = 1
    session: str | None = None


@dataclass(frozen=True)
class _PseudoRipple:
    start: float
    end: float


def run_clusterless_position_validation(
    root: str | Path,
    config: ClusterlessPositionValidationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = ClusterlessPositionValidationConfig() if config is None else config
    rows: list[dict[str, object]] = []
    for session in load_open_field_sessions(root):
        if config.session is not None and session.session_id != config.session:
            continue
        rows.extend(validate_session_clusterless_position(session, config).to_dict("records"))
    samples = pd.DataFrame(rows)
    return samples, summarize_clusterless_position_validation(samples)


def validate_session_clusterless_position(
    session: ReplaySession,
    config: ClusterlessPositionValidationConfig | None = None,
) -> pd.DataFrame:
    """Decode held-out behavior windows with the clusterless mark encoder."""

    config = ClusterlessPositionValidationConfig() if config is None else config
    n_folds = _positive_integer(config.n_folds, "n_folds")
    decode_bin_s = float(config.decode_bin_s)
    if not np.isfinite(decode_bin_s) or decode_bin_s <= 0.0:
        raise ValueError("decode_bin_s must be positive and finite")

    position = _clean_position(session.position)
    if position.size == 0:
        return pd.DataFrame()
    times = position[:, 0]
    xy = position[:, 1:3]
    speed = _speed_cm_s(times, xy)
    encoding_config = config.clusterless.encoding
    min_speed = 5.0 if encoding_config is None else encoding_config.min_speed_cm_s
    base_run_times = _base_run_intervals(session.run_times, times)
    movement = _times_in_intervals(times, base_run_times) & (speed >= min_speed)
    windows = _decode_windows(times, xy, movement, base_run_times, decode_bin_s)

    rng = np.random.default_rng(config.random_seed)
    if config.max_windows_per_session is not None and len(windows) > config.max_windows_per_session:
        keep = np.sort(rng.choice(len(windows), size=config.max_windows_per_session, replace=False))
        windows = [windows[int(index)] for index in keep]
    if not windows:
        return pd.DataFrame()

    shuffled = rng.permutation(len(windows))
    folds = [fold for fold in np.array_split(shuffled, min(n_folds, len(windows))) if fold.size]
    rows: list[dict[str, object]] = []
    for fold_index, validation_indices in enumerate(folds):
        held_out_intervals = np.asarray(
            [
                [windows[int(index)]["start_time"], windows[int(index)]["end_time"]]
                for index in validation_indices
            ],
            dtype=float,
        )
        training_run_times = _subtract_half_open_intervals(base_run_times, held_out_intervals)
        if training_run_times.size == 0:
            continue
        training_session = replace(session, run_times=base_run_times)
        encoding = fit_clusterless_mark_encoding_excluding_intervals(
            training_session,
            config.clusterless,
            held_out_intervals,
        )
        for window_index in sorted(int(index) for index in validation_indices):
            rows.append(
                _decode_clusterless_window(
                    session,
                    encoding,
                    windows[window_index],
                    window_index,
                    fold_index=fold_index,
                )
            )
    return pd.DataFrame(rows)


def _decode_clusterless_window(
    session: ReplaySession,
    encoding,
    window: dict[str, float],
    window_index: int,
    *,
    fold_index: int,
) -> dict[str, object]:
    event = _PseudoRipple(start=float(window["start_time"]), end=float(window["end_time"]))
    emissions = build_clusterless_mark_emissions(
        session,
        encoding,
        event,
        EmissionConfig(time_bin_s=max(event.end - event.start, np.finfo(float).eps)),
    )
    log_likelihood = np.sum(emissions.log_likelihood, axis=0)
    log_posterior = log_likelihood - logsumexp(log_likelihood)
    posterior = np.exp(log_posterior)
    posterior_mean = posterior @ encoding.bin_centers
    map_bin = int(np.argmax(log_posterior))
    true_xy = np.array([float(window["true_x"]), float(window["true_y"])])
    true_bin = int(_nearest_bin(true_xy, encoding.bin_centers))
    true_prob = float(posterior[true_bin])
    true_rank = 1 + int(np.sum(posterior > posterior[true_bin]))
    return {
        "session": session.session_id,
        "fold": int(fold_index),
        "window_index": int(window_index),
        "start_time": float(window["start_time"]),
        "end_time": float(window["end_time"]),
        "center_time": float(window["center_time"]),
        "true_x": float(true_xy[0]),
        "true_y": float(true_xy[1]),
        "posterior_mean_x": float(posterior_mean[0]),
        "posterior_mean_y": float(posterior_mean[1]),
        "map_x": float(encoding.bin_centers[map_bin, 0]),
        "map_y": float(encoding.bin_centers[map_bin, 1]),
        "map_bin": map_bin,
        "true_bin": true_bin,
        "posterior_mean_error_cm": _distance(posterior_mean, true_xy),
        "map_error_cm": _distance(encoding.bin_centers[map_bin], true_xy),
        "true_bin_probability": true_prob,
        "true_bin_rank": true_rank,
        "posterior_entropy": float(-np.sum(posterior * log_posterior)),
        "n_spikes": int(emissions.n_spikes),
        "n_position_bins": int(encoding.n_bins),
        "observation_model": "clusterless-marked-point-process",
        "clusterless_mark_likelihood": str(encoding.mark_likelihood),
        "spike_mark_source": str(encoding.spike_mark_source),
        "spike_mark_features": int(encoding.n_features),
    }


def summarize_clusterless_position_validation(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "decode_windows": ("window_index", "count"),
        "median_posterior_mean_error_cm": ("posterior_mean_error_cm", "median"),
        "median_map_error_cm": ("map_error_cm", "median"),
        "mean_true_bin_probability": ("true_bin_probability", "mean"),
        "median_true_bin_rank": ("true_bin_rank", "median"),
        "mean_spikes_per_window": ("n_spikes", "mean"),
        "spatial_bins": ("n_position_bins", "first"),
        "spike_mark_features": ("spike_mark_features", "first"),
        "clusterless_mark_likelihood": ("clusterless_mark_likelihood", "first"),
    }
    if "fold" in samples.columns:
        aggregations["folds"] = ("fold", "nunique")
    return samples.groupby("session", as_index=False).agg(**aggregations)


def _base_run_intervals(run_times: np.ndarray, position_times: np.ndarray) -> np.ndarray:
    intervals = np.asarray(run_times, dtype=float)
    if intervals.size:
        return np.atleast_2d(intervals)
    return np.asarray([[float(position_times[0]), float(position_times[-1])]], dtype=float)


def _subtract_half_open_intervals(base_intervals: np.ndarray, excluded_intervals: np.ndarray) -> np.ndarray:
    """Subtract half-open validation windows from inclusive run intervals."""

    excluded = _merge_half_open_intervals(excluded_intervals)
    output: list[list[float]] = []
    for raw_start, raw_end in np.asarray(base_intervals, dtype=float):
        start = float(raw_start)
        end = float(raw_end)
        if not np.isfinite(start) or not np.isfinite(end) or end < start:
            continue
        cursor = start
        for excluded_start, excluded_end in excluded:
            if excluded_end <= cursor or excluded_start > end:
                continue
            if excluded_start > cursor:
                training_end = float(np.nextafter(excluded_start, -np.inf))
                if training_end >= cursor:
                    output.append([cursor, min(training_end, end)])
            cursor = max(cursor, float(excluded_end))
            if cursor > end:
                break
        if cursor <= end:
            output.append([cursor, end])
    return np.asarray(output, dtype=float).reshape(-1, 2)


def _merge_half_open_intervals(intervals: np.ndarray) -> list[tuple[float, float]]:
    valid = [
        (float(start), float(end))
        for start, end in np.asarray(intervals, dtype=float).reshape(-1, 2)
        if np.isfinite(start) and np.isfinite(end) and end > start
    ]
    if not valid:
        return []
    valid.sort()
    merged = [valid[0]]
    for start, end in valid[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 1.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _nearest_bin(point: np.ndarray, centers: np.ndarray) -> int:
    return int(np.argmin(np.sum((centers - point[None, :]) ** 2, axis=1)))
