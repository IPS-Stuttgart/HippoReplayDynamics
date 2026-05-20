"""Clusterless behavioral position-decoding validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .clusterless import ClusterlessMarkConfig, build_clusterless_mark_emissions, fit_clusterless_mark_encoding
from .data import ReplaySession, load_open_field_sessions
from .encoding import EmissionConfig, _clean_position, _speed_cm_s, _times_in_intervals
from .position_validation import _decode_windows, _distance


@dataclass(frozen=True)
class ClusterlessPositionValidationConfig:
    clusterless: ClusterlessMarkConfig = ClusterlessMarkConfig()
    decode_bin_s: float = 1.0
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
    """Decode behavior windows with the clusterless marked-point-process encoder."""

    config = ClusterlessPositionValidationConfig() if config is None else config
    position = _clean_position(session.position)
    if position.size == 0:
        return pd.DataFrame()
    times = position[:, 0]
    xy = position[:, 1:3]
    speed = _speed_cm_s(times, xy)
    encoding_config = config.clusterless.encoding
    min_speed = 5.0 if encoding_config is None else encoding_config.min_speed_cm_s
    movement = _times_in_intervals(times, session.run_times) & (speed >= min_speed)
    windows = _decode_windows(times, xy, movement, session.run_times, config.decode_bin_s)
    if config.max_windows_per_session is not None:
        rng = np.random.default_rng(config.random_seed)
        if len(windows) > config.max_windows_per_session:
            keep = np.sort(rng.choice(len(windows), size=config.max_windows_per_session, replace=False))
            windows = [windows[int(index)] for index in keep]
    if not windows:
        return pd.DataFrame()
    encoding = fit_clusterless_mark_encoding(session, config.clusterless)
    rows = []
    for window_index, window in enumerate(windows):
        rows.append(_decode_clusterless_window(session, encoding, window, window_index))
    return pd.DataFrame(rows)


def _decode_clusterless_window(session: ReplaySession, encoding, window: dict[str, float], window_index: int) -> dict[str, object]:
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
        "window_index": int(window_index),
        "start_time": float(window["start_time"]),
        "end_time": float(window["end_time"]),
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
    return samples.groupby("session", as_index=False).agg(
        decode_windows=("window_index", "count"),
        median_posterior_mean_error_cm=("posterior_mean_error_cm", "median"),
        median_map_error_cm=("map_error_cm", "median"),
        mean_true_bin_probability=("true_bin_probability", "mean"),
        median_true_bin_rank=("true_bin_rank", "median"),
        mean_spikes_per_window=("n_spikes", "mean"),
        spatial_bins=("n_position_bins", "first"),
        spike_mark_features=("spike_mark_features", "first"),
        clusterless_mark_likelihood=("clusterless_mark_likelihood", "first"),
    )


def _nearest_bin(point: np.ndarray, centers: np.ndarray) -> int:
    return int(np.argmin(np.sum((centers - point[None, :]) ** 2, axis=1)))
