#!/usr/bin/env python3
"""Track1 decoder QC entry point with enforced unit gates and fold isolation."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

_impl = importlib.import_module("summarize_olafsdottir_track1_decoder_qc_impl")

for _name, _value in vars(_impl).items():
    if _name not in {"__name__", "__file__", "__package__", "__loader__", "__spec__"}:
        globals()[_name] = _value

_original_unit_qc_table = _impl.unit_qc_table


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


def _spikes_in_intervals(spikes, intervals: list[tuple[float, float]]):
    """Return only spikes contained in the half-open training intervals."""

    keep = _impl.sample_mask_in_intervals(spikes.spike_times_s, intervals)
    return _impl.TrackSpikes(
        spike_times_s=spikes.spike_times_s[keep],
        unit_ids=spikes.unit_ids[keep],
        units=tuple(spikes.units),
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
) -> dict[str, object]:
    """Cross-validate without exposing held-out spikes to place-field fitting."""

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
    if windows.empty or not unit_ids:
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
        rates = _impl.fit_place_fields(
            linear=linear,
            times=times,
            valid=train_sample_mask,
            spikes=training_spikes,
            unit_ids=unit_ids,
            edges=edges,
            smoothing_bins=smoothing_bins,
        )
        prior = _impl.occupancy_seconds(linear, times, train_sample_mask, edges)
        prior = (prior + 1e-6) / float(np.nansum(prior + 1e-6))
        for row_index in fold:
            counts = _impl.spike_counts_for_window(
                spikes,
                unit_ids,
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
_impl.crossval_decode = crossval_decode


if __name__ == "__main__":
    raise SystemExit(_impl.main())
