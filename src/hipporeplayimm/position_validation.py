"""Cross-validated behavioral position decoding diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln, logsumexp

from .data import ReplaySession, load_open_field_sessions
from .encoding import (
    EncodingConfig,
    EncodingModel,
    _cell_id_row_indices,
    _clean_position,
    _frame_durations,
    _interp_positions,
    _make_grid,
    _positions_to_flat_bins,
    _speed_cm_s,
    _spikes_and_cell_ids_for_encoding,
    _times_in_intervals,
    _validate_encoding_config,
)

VALIDATED_POSITION_DECODE_BIN_S = 1.0
VALIDATED_POSITION_BIN_SIZE_CM = 6.0
VALIDATED_POSITION_SMOOTHING_SIGMA_BINS = 2.0
VALIDATED_POSITION_MIN_SPEED_CM_S = 5.0


def validated_position_encoding_config() -> EncodingConfig:
    """Return behavior-decoding encoder settings validated on Rat3/Open1-2."""

    return EncodingConfig(
        bin_size_cm=VALIDATED_POSITION_BIN_SIZE_CM,
        smoothing_sigma_bins=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
        min_speed_cm_s=VALIDATED_POSITION_MIN_SPEED_CM_S,
    )


@dataclass(frozen=True)
class PositionDecodingConfig:
    """Configuration for behavioral position-decoding validation."""

    encoding: EncodingConfig = field(default_factory=validated_position_encoding_config)
    decode_bin_s: float = VALIDATED_POSITION_DECODE_BIN_S
    n_folds: int = 5
    max_windows_per_session: int | None = None
    random_seed: int = 1
    min_spikes_per_window: int = 0
    session: str | None = None


@dataclass
class PositionDecodingResult:
    samples: pd.DataFrame
    summary: pd.DataFrame

    def write(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.samples.to_csv(output / "position_decoding_samples.csv", index=False)
        self.samples.to_csv(output / "position_decoding_scores.csv", index=False)
        self.summary.to_csv(output / "position_decoding_summary.csv", index=False)


def run_position_decoding_validation(
    root: str | Path,
    config: PositionDecodingConfig | None = None,
) -> PositionDecodingResult:
    """Validate sorted-spike Poisson position decoding on running behavior."""

    config = PositionDecodingConfig() if config is None else config
    rows: list[dict[str, object]] = []
    sessions = load_open_field_sessions(root)
    for session in sessions:
        if config.session is not None and session.session_id != config.session:
            continue
        rows.extend(validate_session_position_decoding(session, config).to_dict("records"))
    samples = pd.DataFrame(rows)
    return PositionDecodingResult(samples=samples, summary=summarize_position_decoding(samples))


def validate_session_position_decoding(
    session: ReplaySession,
    config: PositionDecodingConfig | None = None,
) -> pd.DataFrame:
    """Return window-level cross-validated position-decoding rows for one session."""

    config = PositionDecodingConfig() if config is None else config
    if config.decode_bin_s <= 0.0:
        raise ValueError("decode_bin_s must be positive")
    if config.n_folds <= 0:
        raise ValueError("n_folds must be positive")

    position = _clean_position(session.position)
    if position.size == 0:
        return pd.DataFrame()
    times = position[:, 0]
    xy = position[:, 1:3]
    speed = _speed_cm_s(times, xy)
    in_run = _times_in_intervals(times, session.run_times)
    movement = in_run & (speed >= config.encoding.min_speed_cm_s)
    windows = _decode_windows(times, xy, movement, session.run_times, config.decode_bin_s)
    rng = np.random.default_rng(config.random_seed)
    windows = _subsample_position_windows(windows, config.max_windows_per_session, rng)
    if not windows:
        return pd.DataFrame()

    shuffled = rng.permutation(len(windows))
    n_folds = min(config.n_folds, len(windows))
    folds = [fold for fold in np.array_split(shuffled, n_folds) if fold.size]
    rows: list[dict[str, object]] = []
    for fold_index, validation_indices in enumerate(folds):
        intervals = np.asarray(
            [
                [windows[int(index)]["start_time"], windows[int(index)]["end_time"]]
                for index in validation_indices
            ],
            dtype=float,
        )
        train_mask = movement & ~_times_in_intervals(times, intervals)
        if not np.any(train_mask):
            continue
        encoding = fit_place_field_encoding_for_position_mask(session, train_mask, config.encoding)
        for window_index in sorted(int(index) for index in validation_indices):
            row = _decode_window(
                session,
                encoding,
                windows[window_index],
                fold_index=fold_index,
                window_index=window_index,
                min_spikes=config.min_spikes_per_window,
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def _subsample_position_windows(
    windows: list[dict[str, float]],
    max_windows: int | None,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Return a seeded, chronologically ordered subset of decode windows."""

    if max_windows is None or len(windows) <= max_windows:
        return windows
    keep = np.sort(rng.choice(len(windows), size=max_windows, replace=False))
    return [windows[int(index)] for index in keep]


def fit_place_field_encoding_for_position_mask(
    session: ReplaySession,
    train_frame_mask: np.ndarray,
    config: EncodingConfig | None = None,
) -> EncodingModel:
    """Fit place fields from an explicit training mask over position frames."""

    config = EncodingConfig() if config is None else config
    _validate_encoding_config(config)
    position = _clean_position(session.position)
    times = position[:, 0]
    xy = position[:, 1:3]
    if train_frame_mask.shape[0] != times.shape[0]:
        raise ValueError("train_frame_mask must have one value per cleaned position frame")

    x_edges, y_edges, centers = _make_grid(xy, config)
    grid_shape = (len(x_edges) - 1, len(y_edges) - 1)
    flat_bins = _positions_to_flat_bins(xy, x_edges, y_edges)
    dt = _frame_durations(times)
    train_frames = np.asarray(train_frame_mask, dtype=bool) & (flat_bins >= 0)

    occupancy = np.zeros(grid_shape[0] * grid_shape[1], dtype=float)
    np.add.at(occupancy, flat_bins[train_frames], dt[train_frames])

    spikes, cell_ids = _spikes_and_cell_ids_for_encoding(session, config)
    cell_ids = np.asarray(sorted(np.unique(cell_ids)), dtype=int)
    counts = np.zeros((cell_ids.shape[0], occupancy.shape[0]), dtype=float)

    if spikes.size and cell_ids.size:
        spike_times = spikes[:, 0]
        spike_cell_ids = spikes[:, 1].astype(int)
        spike_xy = _interp_positions(times, xy, spike_times)
        spike_bins = _positions_to_flat_bins(spike_xy, x_edges, y_edges)
        frame_indices = np.searchsorted(times, spike_times, side="right") - 1
        valid_frames = (frame_indices >= 0) & (frame_indices < times.shape[0])
        spike_in_training = np.zeros(spike_times.shape, dtype=bool)
        if np.any(valid_frames):
            rows_for_spikes = frame_indices[valid_frames].astype(int)
            offsets = spike_times[valid_frames] - times[rows_for_spikes]
            spike_in_training[valid_frames] = (
                train_frames[rows_for_spikes]
                & (offsets >= 0.0)
                & (offsets < dt[rows_for_spikes])
            )
        keep_spikes = spike_in_training & (spike_bins >= 0)
        kept_cell_ids = spike_cell_ids[keep_spikes]
        kept_bins = spike_bins[keep_spikes].astype(int)
        rows = np.searchsorted(cell_ids, kept_cell_ids)
        valid_rows = (rows >= 0) & (rows < cell_ids.shape[0])
        valid_rows[valid_rows] &= cell_ids[rows[valid_rows]] == kept_cell_ids[valid_rows]
        np.add.at(counts, (rows[valid_rows], kept_bins[valid_rows]), 1.0)

    occupancy_grid = occupancy.reshape(grid_shape)
    if config.smoothing_sigma_bins > 0.0:
        smooth_occupancy = gaussian_filter(occupancy_grid, sigma=config.smoothing_sigma_bins, mode="constant").reshape(-1)
        smooth_counts = np.vstack(
            [
                gaussian_filter(row.reshape(grid_shape), sigma=config.smoothing_sigma_bins, mode="constant").reshape(-1)
                for row in counts
            ]
        ) if counts.shape[0] else counts
    else:
        smooth_occupancy = occupancy
        smooth_counts = counts

    denominator = np.maximum(smooth_occupancy, config.min_occupancy_s)
    rates = smooth_counts / denominator[None, :] if smooth_counts.shape[0] else smooth_counts
    rates = np.maximum(rates, config.rate_floor_hz)
    return EncodingModel(
        x_edges=x_edges,
        y_edges=y_edges,
        bin_centers=centers,
        rates_hz=rates,
        occupancy_s=occupancy,
        cell_ids=cell_ids,
        config=config,
    )


def summarize_position_decoding(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    return (
        samples.groupby("session", as_index=False)
        .agg(
            decode_windows=("window_index", "count"),
            folds=("fold", "nunique"),
            median_posterior_mean_error_cm=("posterior_mean_error_cm", "median"),
            median_map_error_cm=("map_error_cm", "median"),
            mean_true_bin_probability=("true_bin_probability", "mean"),
            median_true_bin_rank=("true_bin_rank", "median"),
            mean_spikes_per_window=("n_spikes", "mean"),
            cells=("n_cells", "first"),
            spatial_bins=("n_position_bins", "first"),
            spike_mark_features=("spike_mark_features", "first"),
            clusterless_mark_likelihood=("clusterless_mark_likelihood", "first"),
        )
        .sort_values("session")
    )


def _decode_windows(
    times: np.ndarray,
    xy: np.ndarray,
    movement: np.ndarray,
    run_times: np.ndarray,
    decode_bin_s: float,
) -> list[dict[str, float]]:
    intervals = run_times if run_times.size else np.array([[times[0], times[-1]]], dtype=float)
    windows: list[dict[str, float]] = []
    for start, end in np.asarray(intervals, dtype=float):
        if end <= start:
            continue
        edges = np.arange(start, end, decode_bin_s)
        if edges.size == 0 or edges[-1] < end:
            edges = np.append(edges, end)
        for left, right in zip(edges[:-1], edges[1:], strict=True):
            keep = (times >= left) & (times < right) & movement
            if not np.any(keep):
                continue
            true_xy = np.median(xy[keep], axis=0)
            windows.append(
                {
                    "start_time": float(left),
                    "end_time": float(right),
                    "center_time": float(0.5 * (left + right)),
                    "true_x": float(true_xy[0]),
                    "true_y": float(true_xy[1]),
                }
            )
    return windows


def _decode_window(
    session: ReplaySession,
    encoding: EncodingModel,
    window: dict[str, float],
    *,
    fold_index: int,
    window_index: int,
    min_spikes: int,
) -> dict[str, object] | None:
    start = float(window["start_time"])
    end = float(window["end_time"])
    dt = max(end - start, np.finfo(float).eps)
    counts = _spike_counts_for_window(session, encoding, start, end)
    n_spikes = int(counts.sum())
    if n_spikes < min_spikes:
        return None

    expected = encoding.rates_hz * dt
    if encoding.n_cells:
        log_expected = np.log(expected)
        log_likelihood = counts @ log_expected - expected.sum(axis=0)
        log_likelihood -= float(gammaln(counts + 1).sum())
    else:
        log_likelihood = np.zeros(encoding.n_bins, dtype=float)
    log_posterior = log_likelihood - logsumexp(log_likelihood)
    posterior = np.exp(log_posterior)
    posterior_mean = posterior @ encoding.bin_centers
    map_bin = int(np.argmax(log_posterior))
    true_xy = np.array([[float(window["true_x"]), float(window["true_y"])]] )
    true_bin = int(encoding.positions_to_flat_bins(true_xy)[0])
    if true_bin >= 0:
        true_prob = float(posterior[true_bin])
        true_rank = 1 + int(np.sum(posterior > posterior[true_bin]))
    else:
        true_prob = np.nan
        true_rank = np.nan

    marks = session.spike_marks
    return {
        "session": session.session_id,
        "fold": int(fold_index),
        "window_index": int(window_index),
        "start_time": start,
        "end_time": end,
        "center_time": float(window["center_time"]),
        "true_x": float(window["true_x"]),
        "true_y": float(window["true_y"]),
        "posterior_mean_x": float(posterior_mean[0]),
        "posterior_mean_y": float(posterior_mean[1]),
        "map_x": float(encoding.bin_centers[map_bin, 0]),
        "map_y": float(encoding.bin_centers[map_bin, 1]),
        "map_bin": map_bin,
        "true_bin": true_bin,
        "posterior_mean_error_cm": _distance(posterior_mean, true_xy[0]),
        "map_error_cm": _distance(encoding.bin_centers[map_bin], true_xy[0]),
        "true_bin_probability": true_prob,
        "true_bin_rank": true_rank,
        "posterior_entropy": float(-np.sum(posterior * log_posterior)),
        "n_spikes": n_spikes,
        "n_cells": int(encoding.n_cells),
        "n_position_bins": int(encoding.n_bins),
        "observation_model": "sorted-spike-poisson",
        "spike_mark_features": 0 if marks is None else marks.n_features,
        "spike_mark_source": "" if marks is None else f"{marks.source_file}:{marks.source_variable}",
        "clusterless_mark_likelihood": "not_implemented",
    }


def _spike_counts_for_window(
    session: ReplaySession,
    encoding: EncodingModel,
    start: float,
    end: float,
) -> np.ndarray:
    counts = np.zeros(encoding.n_cells, dtype=int)
    spikes, _ = _spikes_and_cell_ids_for_encoding(session, encoding.config)
    if not spikes.size or not encoding.n_cells:
        return counts
    spike_cell_ids = spikes[:, 1].astype(int)
    keep = (spikes[:, 0] >= start) & (spikes[:, 0] < end) & np.isin(spike_cell_ids, encoding.cell_ids)
    rows = _cell_id_row_indices(encoding.cell_ids, spike_cell_ids[keep])
    valid = rows >= 0
    np.add.at(counts, rows[valid], 1)
    return counts


def _mask_to_intervals(times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not np.any(mask):
        return np.empty((0, 2), dtype=float)
    durations = _frame_durations(times)
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    intervals = []
    for start, stop in zip(changes[0::2], changes[1::2], strict=True):
        intervals.append([float(times[start]), float(times[stop - 1] + durations[stop - 1])])
    return np.asarray(intervals, dtype=float)


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return float(np.sqrt(np.sum(delta * delta)))
