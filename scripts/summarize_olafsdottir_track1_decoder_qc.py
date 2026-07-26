#!/usr/bin/env python3
"""Track1 decoder QC entry point with enforced unit gates and nested fold isolation."""

from __future__ import annotations

from contextvars import ContextVar
import importlib

import numpy as np
import pandas as pd

_impl = importlib.import_module("summarize_olafsdottir_track1_decoder_qc_impl")

for _name, _value in vars(_impl).items():
    if _name not in {"__name__", "__file__", "__package__", "__loader__", "__spec__"}:
        globals()[_name] = _value

_original_unit_qc_table = _impl.unit_qc_table
_original_summarize_pair = _impl.summarize_pair
_crossval_unit_qc_context: ContextVar[dict[str, float | int] | None] = ContextVar(
    "track1_crossval_unit_qc_context",
    default=None,
)


def unit_qc_table(
    *,
    animal: str,
    date: str,
    track_session: str,
    linearized: pd.DataFrame,
    spikes,
    position_bin_size_cm: float,
    min_unit_spikes: int,
    min_unit_mean_rate_hz: float,
    min_place_information_bits: float,
    min_place_peak_rate_hz: float,
    smoothing_bins: int,
):
    """Build unit QC while enforcing both configured place-field thresholds."""

    table, place_fields = _original_unit_qc_table(
        animal=animal,
        date=date,
        track_session=track_session,
        linearized=linearized,
        spikes=spikes,
        position_bin_size_cm=position_bin_size_cm,
        min_unit_spikes=min_unit_spikes,
        min_unit_mean_rate_hz=min_unit_mean_rate_hz,
        min_place_information_bits=min_place_information_bits,
        min_place_peak_rate_hz=min_place_peak_rate_hz,
        smoothing_bins=smoothing_bins,
    )
    basic_qc = table["unit_qc_passed"].map(_impl._as_bool)
    spatial_information_values = pd.to_numeric(
        table["spatial_information"], errors="coerce"
    )
    peak_rates = pd.to_numeric(table["peak_rate_hz"], errors="coerce")
    place_like = (
        basic_qc
        & (spatial_information_values >= float(min_place_information_bits))
        & (peak_rates >= float(min_place_peak_rate_hz))
    )
    table.loc[:, "unit_qc_passed"] = place_like.astype(bool)
    table.attrs["n_place_like_units"] = int(place_like.sum())
    return table, place_fields


def summarize_pair(
    pair: pd.Series,
    *,
    dataset_root,
    linearization: pd.DataFrame,
    linearization_root,
    output_dir,
    min_encoding_units: int,
    crossval_folds: int,
    position_bin_size_cm: float,
    decode_window_s: float,
    min_unit_spikes: int,
    min_unit_mean_rate_hz: float,
    min_place_information_bits: float,
    min_place_peak_rate_hz: float,
    smoothing_bins: int,
    max_posterior_median_error_cm: float,
    max_map_median_error_cm: float,
    min_posterior_coverage_fraction: float,
):
    """Run one session with fold-local unit selection thresholds."""

    token = _crossval_unit_qc_context.set(
        {
            "min_unit_spikes": int(min_unit_spikes),
            "min_unit_mean_rate_hz": float(min_unit_mean_rate_hz),
            "min_place_information_bits": float(min_place_information_bits),
            "min_place_peak_rate_hz": float(min_place_peak_rate_hz),
        }
    )
    try:
        return _original_summarize_pair(
            pair,
            dataset_root=dataset_root,
            linearization=linearization,
            linearization_root=linearization_root,
            output_dir=output_dir,
            min_encoding_units=min_encoding_units,
            crossval_folds=crossval_folds,
            position_bin_size_cm=position_bin_size_cm,
            decode_window_s=decode_window_s,
            min_unit_spikes=min_unit_spikes,
            min_unit_mean_rate_hz=min_unit_mean_rate_hz,
            min_place_information_bits=min_place_information_bits,
            min_place_peak_rate_hz=min_place_peak_rate_hz,
            smoothing_bins=smoothing_bins,
            max_posterior_median_error_cm=max_posterior_median_error_cm,
            max_map_median_error_cm=max_map_median_error_cm,
            min_posterior_coverage_fraction=min_posterior_coverage_fraction,
        )
    finally:
        _crossval_unit_qc_context.reset(token)


def _spikes_in_intervals(spikes, intervals: list[tuple[float, float]]):
    """Return only spikes contained in the half-open training intervals."""

    keep = _impl.sample_mask_in_intervals(spikes.spike_times_s, intervals)
    return _impl.TrackSpikes(
        spike_times_s=spikes.spike_times_s[keep],
        unit_ids=spikes.unit_ids[keep],
        units=tuple(spikes.units),
    )


def decode_windows(linearized: pd.DataFrame, decode_window_s: float) -> pd.DataFrame:
    """Create decoding windows bounded by the valid position support."""

    window = float(decode_window_s)
    if not np.isfinite(window) or window <= 0.0:
        raise ValueError("decode_window_s must be finite and positive")

    times = pd.to_numeric(
        linearized["time_s"], errors="coerce"
    ).to_numpy(dtype=float)
    linear = pd.to_numeric(
        linearized["linear_position_cm"], errors="coerce"
    ).to_numpy(dtype=float)
    valid = _impl.valid_position_mask(linearized)
    columns = ["start_time_s", "end_time_s", "true_position_cm"]
    if not np.any(valid):
        return pd.DataFrame(columns=columns)

    start = float(np.nanmin(times[valid]))
    end = float(np.nanmax(times[valid]))
    if end <= start:
        return pd.DataFrame(columns=columns)

    duration = end - start
    tolerance = 16.0 * np.finfo(float).eps * max(
        abs(start), abs(end), abs(duration), 1.0
    )
    closed_end = float(np.nextafter(end, np.inf))
    edges = np.arange(start, end, window, dtype=float)
    if edges.size == 0:
        edges = np.asarray([start], dtype=float)
    if edges[-1] < end - tolerance:
        edges = np.append(edges, closed_end)
    else:
        edges[-1] = closed_end

    rows: list[dict[str, float]] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        keep = valid & (times >= left) & (times < right)
        if np.any(keep):
            rows.append(
                {
                    "start_time_s": float(left),
                    "end_time_s": float(right),
                    "true_position_cm": float(np.nanmedian(linear[keep])),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _interval_sample_durations(
    times: np.ndarray,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    """Clip each sample's forward duration to the training intervals."""

    times = np.asarray(times, dtype=float)
    durations = np.zeros(times.shape, dtype=float)
    if times.size == 0 or not intervals:
        return durations

    base_durations = _impl.sample_durations(times)
    valid = (
        np.isfinite(times)
        & np.isfinite(base_durations)
        & (base_durations > 0.0)
    )
    segment_ends = times + np.where(valid, base_durations, 0.0)
    for start, end in intervals:
        left = float(start)
        right = float(end)
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            continue
        overlap = np.minimum(segment_ends, right) - np.maximum(times, left)
        durations += np.where(valid, np.maximum(overlap, 0.0), 0.0)
    return durations


def _occupancy_seconds_in_intervals(
    linear: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    edges: np.ndarray,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    """Accumulate occupancy without assigning held-out time to training samples."""

    durations = _interval_sample_durations(times, intervals)
    keep = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(linear)
        & np.isfinite(durations)
        & (durations > 0.0)
    )
    bins = np.searchsorted(edges, linear[keep], side="right") - 1
    bins = np.clip(bins, 0, edges.shape[0] - 2)
    occupancy = np.zeros(edges.shape[0] - 1, dtype=float)
    np.add.at(occupancy, bins, durations[keep])
    return occupancy


def _fit_place_fields_with_occupancy(
    *,
    linear: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    spikes,
    unit_ids: tuple[int, ...],
    edges: np.ndarray,
    smoothing_bins: int,
    occupancy: np.ndarray,
) -> np.ndarray:
    """Fit rates using the same fold-isolated occupancy as the decoder prior."""

    fields = np.zeros((len(unit_ids), edges.shape[0] - 1), dtype=float)
    for unit_index, unit_id in enumerate(unit_ids):
        unit_times = spikes.spike_times_s[spikes.unit_ids == int(unit_id)]
        spike_pos = _impl.interpolate_position_at_times(
            unit_times,
            times,
            linear,
            valid,
        )
        counts, _ = np.histogram(spike_pos[np.isfinite(spike_pos)], bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = counts / occupancy
        rates[~np.isfinite(rates)] = 0.0
        fields[unit_index, :] = _impl.smooth_1d(rates, smoothing_bins)
    return fields


def _select_fold_units(
    *,
    candidate_unit_ids: tuple[int, ...],
    rates: np.ndarray,
    occupancy: np.ndarray,
    training_spikes,
    min_unit_spikes: int,
    min_unit_mean_rate_hz: float,
    min_place_information_bits: float,
    min_place_peak_rate_hz: float,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Apply every configured unit gate using training data only."""

    training_duration_s = float(np.nansum(occupancy))
    selected_indices: list[int] = []
    selected_ids: list[int] = []
    for index, unit_id in enumerate(candidate_unit_ids):
        n_spikes = int(
            np.count_nonzero(training_spikes.unit_ids == int(unit_id))
        )
        mean_rate_hz = (
            float(n_spikes / training_duration_s)
            if training_duration_s > 0.0
            else np.nan
        )
        unit_rates = np.asarray(rates[index], dtype=float)
        peak_rate_hz = (
            float(np.nanmax(unit_rates))
            if unit_rates.size and np.isfinite(unit_rates).any()
            else np.nan
        )
        information = _impl.spatial_information(unit_rates, occupancy)
        passed = (
            n_spikes >= int(min_unit_spikes)
            and np.isfinite(mean_rate_hz)
            and mean_rate_hz >= float(min_unit_mean_rate_hz)
            and np.isfinite(peak_rate_hz)
            and peak_rate_hz > 0.0
            and peak_rate_hz >= float(min_place_peak_rate_hz)
            and np.isfinite(information)
            and information >= float(min_place_information_bits)
        )
        if passed:
            selected_indices.append(index)
            selected_ids.append(int(unit_id))

    if not selected_indices:
        return (), np.empty((0, rates.shape[1]), dtype=float)
    return (
        tuple(selected_ids),
        np.asarray(rates[np.asarray(selected_indices, dtype=int)], dtype=float),
    )


def crossval_decode(
    *,
    linearized: pd.DataFrame,
    spikes,
    unit_ids: tuple[int, ...],
    crossval_folds: int,
    position_bin_size_cm: float,
    decode_window_s: float,
    smoothing_bins: int,
    min_unit_spikes: int = 1,
    min_unit_mean_rate_hz: float = 0.0,
    min_place_information_bits: float = 0.0,
    min_place_peak_rate_hz: float = 0.0,
) -> dict[str, object]:
    """Cross-validate place fields, occupancy, and unit selection by training fold."""

    context = _crossval_unit_qc_context.get()
    if context is not None:
        candidate_unit_ids = tuple(int(unit_id) for unit_id in spikes.units)
        min_unit_spikes = int(context["min_unit_spikes"])
        min_unit_mean_rate_hz = float(context["min_unit_mean_rate_hz"])
        min_place_information_bits = float(context["min_place_information_bits"])
        min_place_peak_rate_hz = float(context["min_place_peak_rate_hz"])
    else:
        candidate_unit_ids = tuple(int(unit_id) for unit_id in unit_ids)

    windows = _impl.decode_windows(linearized, decode_window_s)
    empty = {
        "crossval_n_folds": int(crossval_folds),
        "posterior_mean_error_cm_median": np.nan,
        "posterior_mean_error_cm_p75": np.nan,
        "posterior_mean_error_cm_p90": np.nan,
        "map_error_cm_median": np.nan,
        "map_error_cm_p75": np.nan,
        "map_error_cm_p90": np.nan,
        "posterior_coverage_fraction": 0.0,
        "true_position_cm": np.asarray([], dtype=float),
        "posterior_mean_position_cm": np.asarray([], dtype=float),
        "map_position_cm": np.asarray([], dtype=float),
    }
    if windows.empty or not candidate_unit_ids:
        return empty

    true_pos = pd.to_numeric(
        windows["true_position_cm"], errors="coerce"
    ).to_numpy(dtype=float)
    starts = pd.to_numeric(windows["start_time_s"], errors="coerce").to_numpy(
        dtype=float
    )
    ends = pd.to_numeric(windows["end_time_s"], errors="coerce").to_numpy(
        dtype=float
    )
    valid_window = (
        np.isfinite(true_pos)
        & np.isfinite(starts)
        & np.isfinite(ends)
        & (ends > starts)
    )
    if valid_window.sum() < max(2, int(crossval_folds)):
        return empty

    indices = np.flatnonzero(valid_window)
    folds = np.array_split(indices, min(int(crossval_folds), indices.shape[0]))
    posterior_predictions = np.full(windows.shape[0], np.nan, dtype=float)
    map_predictions = np.full(windows.shape[0], np.nan, dtype=float)
    linear = pd.to_numeric(
        linearized["linear_position_cm"], errors="coerce"
    ).to_numpy(dtype=float)
    times = pd.to_numeric(linearized["time_s"], errors="coerce").to_numpy(
        dtype=float
    )
    valid_position = _impl.valid_position_mask(linearized)
    edges = _impl.position_edges(linear[valid_position], position_bin_size_cm)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for fold in folds:
        if fold.size == 0:
            continue
        test_mask = np.zeros(windows.shape[0], dtype=bool)
        test_mask[fold] = True
        train_intervals = [
            (float(starts[i]), float(ends[i]))
            for i in indices
            if not test_mask[i]
        ]
        train_sample_mask = (
            _impl.sample_mask_in_intervals(times, train_intervals) & valid_position
        )
        training_spikes = _spikes_in_intervals(spikes, train_intervals)
        occupancy = _occupancy_seconds_in_intervals(
            linear,
            times,
            train_sample_mask,
            edges,
            train_intervals,
        )
        candidate_rates = _fit_place_fields_with_occupancy(
            linear=linear,
            times=times,
            valid=train_sample_mask,
            spikes=training_spikes,
            unit_ids=candidate_unit_ids,
            edges=edges,
            smoothing_bins=smoothing_bins,
            occupancy=occupancy,
        )
        fold_unit_ids, rates = _select_fold_units(
            candidate_unit_ids=candidate_unit_ids,
            rates=candidate_rates,
            occupancy=occupancy,
            training_spikes=training_spikes,
            min_unit_spikes=min_unit_spikes,
            min_unit_mean_rate_hz=min_unit_mean_rate_hz,
            min_place_information_bits=min_place_information_bits,
            min_place_peak_rate_hz=min_place_peak_rate_hz,
        )
        if not fold_unit_ids:
            continue
        prior = (occupancy + 1e-6) / float(np.nansum(occupancy + 1e-6))
        for row_index in fold:
            counts = _impl.spike_counts_for_window(
                spikes,
                fold_unit_ids,
                float(starts[row_index]),
                float(ends[row_index]),
            )
            posterior = _impl.poisson_posterior(
                counts,
                rates,
                float(ends[row_index] - starts[row_index]),
                prior,
            )
            if (
                posterior.size
                and np.isfinite(posterior).all()
                and posterior.sum() > 0.0
            ):
                posterior_predictions[row_index] = float(
                    np.sum(posterior * centers)
                )
                map_predictions[row_index] = float(
                    centers[int(np.argmax(posterior))]
                )

    decoded = (
        valid_window
        & np.isfinite(posterior_predictions)
        & np.isfinite(map_predictions)
    )
    if not np.any(decoded):
        return empty

    posterior_error = np.abs(posterior_predictions[decoded] - true_pos[decoded])
    map_error = np.abs(map_predictions[decoded] - true_pos[decoded])
    return {
        "crossval_n_folds": int(len(folds)),
        "posterior_mean_error_cm_median": _impl.percentile(
            posterior_error, 50.0
        ),
        "posterior_mean_error_cm_p75": _impl.percentile(posterior_error, 75.0),
        "posterior_mean_error_cm_p90": _impl.percentile(posterior_error, 90.0),
        "map_error_cm_median": _impl.percentile(map_error, 50.0),
        "map_error_cm_p75": _impl.percentile(map_error, 75.0),
        "map_error_cm_p90": _impl.percentile(map_error, 90.0),
        "posterior_coverage_fraction": float(
            np.count_nonzero(decoded) / np.count_nonzero(valid_window)
        ),
        "true_position_cm": true_pos[decoded],
        "posterior_mean_position_cm": posterior_predictions[decoded],
        "map_position_cm": map_predictions[decoded],
    }


_impl.unit_qc_table = unit_qc_table
_impl.summarize_pair = summarize_pair
_impl.decode_windows = decode_windows
_impl.crossval_decode = crossval_decode


if __name__ == "__main__":
    raise SystemExit(_impl.main())
