#!/usr/bin/env python3
"""Run Track1 encoding/decoder QC for Olafsdottir2016 1D Z-track pairs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hipporeplayimm.olafsdottir2016 import read_axona_cut
from linearize_olafsdottir_ztrack import linearize_pos_file


UNIT_OUTPUT = "olafsdottir_track1_encoding_unit_qc.csv"
DECODER_OUTPUT = "olafsdottir_track1_decoder_crossval_qc.csv"
ANIMAL_OUTPUT = "olafsdottir_track1_decoder_by_animal.csv"
GATE_OUTPUT = "olafsdottir_track1_decoder_gate_summary.csv"
SUMMARY_OUTPUT = "olafsdottir_track1_decoder_qc_summary.md"
FIGURE_OUTPUT = "olafsdottir_track1_decoder_figures_manifest.csv"

REQUIRED_PAIR_COLUMNS = {
    "animal",
    "date",
    "track_session",
    "sleepPOST_session",
    "hippocampal_tetrodes",
    "usable_pair",
}
REQUIRED_LINEARIZATION_COLUMNS = {
    "animal",
    "date",
    "track_session",
    "sleeppost_session",
    "linearization_status",
    "valid_position_fraction",
    "linearized_position_span_cm",
    "occupancy_nonzero_bins",
    "orientation_rule",
    "reversal_applied",
}

UNIT_COLUMNS = [
    "animal",
    "date",
    "track1_session",
    "unit_id",
    "n_spikes_track1",
    "mean_rate_hz",
    "peak_rate_hz",
    "spatial_information",
    "place_field_peak_cm",
    "place_field_width_cm",
    "occupancy_covered_fraction",
    "unit_qc_passed",
]

DECODER_COLUMNS = [
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "n_units_track1",
    "n_position_samples",
    "track_duration_s",
    "valid_position_fraction",
    "linearized_track_span_cm",
    "occupancy_nonzero_bins",
    "encoding_units_passing_qc",
    "median_firing_rate_hz",
    "n_place_like_units",
    "crossval_n_folds",
    "posterior_mean_error_cm_median",
    "posterior_mean_error_cm_p75",
    "posterior_mean_error_cm_p90",
    "map_error_cm_median",
    "map_error_cm_p75",
    "map_error_cm_p90",
    "posterior_coverage_fraction",
    "decoder_status",
    "decoder_qc_paper_ready",
    "decoder_qc_scoring_available",
    "decoder_scoring_available_reason",
    "orientation_rule",
    "reversal_applied",
    "exclusion_reason",
]


@dataclass(frozen=True)
class TrackSpikes:
    spike_times_s: np.ndarray
    unit_ids: np.ndarray
    units: tuple[int, ...]


@dataclass(frozen=True)
class PlaceFields:
    unit_ids: tuple[int, ...]
    bin_centers_cm: np.ndarray
    occupancy_s: np.ndarray
    rates_hz: np.ndarray
    occupancy_covered_fraction: float


def run_decoder_qc(
    *,
    dataset_root: str | Path,
    pairs_csv: str | Path,
    linearization_qc: str | Path,
    output_dir: str | Path,
    min_encoding_units: int = 5,
    max_posterior_median_error_cm: float = 35.0,
    max_map_median_error_cm: float = 45.0,
    min_posterior_coverage_fraction: float = 0.80,
    crossval_folds: int = 5,
    position_bin_size_cm: float = 5.0,
    decode_window_s: float = 0.250,
    min_unit_spikes: int = 5,
    min_unit_mean_rate_hz: float = 0.01,
    min_place_information_bits: float = 0.05,
    min_place_peak_rate_hz: float = 0.05,
    smoothing_bins: int = 1,
    max_decoder_animal_fraction: float = 0.75,
    max_decoder_session_fraction: float = 0.75,
) -> dict[str, pd.DataFrame]:
    pairs = load_pairs(pairs_csv)
    linearization = load_linearization_qc(linearization_qc)
    usable_pairs = pairs[pairs["usable_pair"].map(_as_bool)].copy()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    decoder_rows: list[dict[str, object]] = []
    unit_tables: list[pd.DataFrame] = []
    figure_rows: list[dict[str, object]] = []
    linearization_root = Path(linearization_qc).resolve().parent
    for _, pair in usable_pairs.iterrows():
        row, units, figures = summarize_pair(
            pair,
            dataset_root=Path(dataset_root),
            linearization=linearization,
            linearization_root=linearization_root,
            output_dir=out / "sessions" / str(pair["animal"]).upper() / str(pair["date"]),
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
        decoder_rows.append(row)
        unit_tables.append(units)
        figure_rows.extend(figures)

    decoder = pd.DataFrame(decoder_rows, columns=DECODER_COLUMNS)
    units = pd.concat(unit_tables, ignore_index=True) if unit_tables else pd.DataFrame(columns=UNIT_COLUMNS)
    animals = summarize_by_animal(decoder, units)
    gates = gate_summary(
        pairs=usable_pairs,
        decoder=decoder,
        min_encoding_units=min_encoding_units,
        max_posterior_median_error_cm=max_posterior_median_error_cm,
        max_map_median_error_cm=max_map_median_error_cm,
        min_posterior_coverage_fraction=min_posterior_coverage_fraction,
        max_decoder_animal_fraction=max_decoder_animal_fraction,
        max_decoder_session_fraction=max_decoder_session_fraction,
    )
    figures = pd.DataFrame(
        figure_rows,
        columns=["animal", "date", "track1_session", "figure_type", "figure_path"],
    )

    units.to_csv(out / UNIT_OUTPUT, index=False)
    decoder.to_csv(out / DECODER_OUTPUT, index=False)
    animals.to_csv(out / ANIMAL_OUTPUT, index=False)
    gates.to_csv(out / GATE_OUTPUT, index=False)
    figures.to_csv(out / FIGURE_OUTPUT, index=False)
    (out / SUMMARY_OUTPUT).write_text(
        build_markdown_summary(
            decoder,
            animals,
            gates,
            min_encoding_units=min_encoding_units,
            max_posterior_median_error_cm=max_posterior_median_error_cm,
            max_map_median_error_cm=max_map_median_error_cm,
            min_posterior_coverage_fraction=min_posterior_coverage_fraction,
            crossval_folds=crossval_folds,
            position_bin_size_cm=position_bin_size_cm,
            decode_window_s=decode_window_s,
        ),
        encoding="utf-8",
    )
    return {
        "units": units,
        "decoder": decoder,
        "animals": animals,
        "gates": gates,
        "figures": figures,
    }


def load_pairs(path: str | Path) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    missing = sorted(REQUIRED_PAIR_COLUMNS.difference(pairs.columns))
    if missing:
        raise ValueError(f"pairs CSV is missing required columns: {missing}")
    return pairs


def load_linearization_qc(path: str | Path) -> pd.DataFrame:
    linearization = pd.read_csv(path)
    missing = sorted(REQUIRED_LINEARIZATION_COLUMNS.difference(linearization.columns))
    if missing:
        raise ValueError(f"linearization QC is missing required columns: {missing}")
    return linearization


def summarize_pair(
    pair: pd.Series,
    *,
    dataset_root: Path,
    linearization: pd.DataFrame,
    linearization_root: Path,
    output_dir: Path,
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
) -> tuple[dict[str, object], pd.DataFrame, list[dict[str, object]]]:
    animal = str(pair["animal"]).upper()
    date = str(pair["date"])
    track_session = str(pair["track_session"])
    sleep_session = str(pair["sleepPOST_session"])
    tetrodes = _parse_tetrodes(str(pair["hippocampal_tetrodes"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    figures: list[dict[str, object]] = []
    lin_row = matching_linearization_row(linearization, animal, date, track_session)
    orientation_rule = str(lin_row.get("orientation_rule", "")) if lin_row is not None else ""
    reversal_applied = bool(_as_bool(lin_row.get("reversal_applied", False))) if lin_row is not None else False

    try:
        if lin_row is None:
            raise ValueError("missing_linearization_qc_row")
        if str(lin_row["linearization_status"]) != "pass":
            raise ValueError("linearization_qc_not_passed")
        linearized = load_or_create_linearized_position(
            dataset_root=dataset_root,
            linearization_root=linearization_root,
            animal=animal,
            date=date,
            track_session=track_session,
            output_dir=output_dir,
            position_bin_size_cm=position_bin_size_cm,
        )
        spikes = load_track_spikes(_track_stem_path(dataset_root, animal, date, track_session), tetrodes)
        unit_table, place_fields = unit_qc_table(
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
        passing_units = unit_table[unit_table["unit_qc_passed"].map(_as_bool)]["unit_id"].astype(int).tolist()
        if len(passing_units) < int(min_encoding_units):
            reasons.append("too_few_encoding_units")
        crossval = crossval_decode(
            linearized=linearized,
            spikes=spikes,
            unit_ids=tuple(passing_units),
            crossval_folds=crossval_folds,
            position_bin_size_cm=position_bin_size_cm,
            decode_window_s=decode_window_s,
            smoothing_bins=smoothing_bins,
        )
        if crossval["posterior_coverage_fraction"] < float(min_posterior_coverage_fraction):
            reasons.append("low_posterior_coverage")
        if not np.isfinite(crossval["posterior_mean_error_cm_median"]):
            reasons.append("missing_posterior_mean_errors")
        if not np.isfinite(crossval["map_error_cm_median"]):
            reasons.append("missing_map_errors")
        if (
            np.isfinite(crossval["posterior_mean_error_cm_median"])
            and crossval["posterior_mean_error_cm_median"] > float(max_posterior_median_error_cm)
        ):
            reasons.append("posterior_median_error_above_threshold")
        if np.isfinite(crossval["map_error_cm_median"]) and crossval["map_error_cm_median"] > float(max_map_median_error_cm):
            reasons.append("map_median_error_above_threshold")
        if animal == "R2142" and not reversal_applied:
            reasons.append("r2142_reversal_not_applied")
        row = decoder_row(
            animal=animal,
            date=date,
            track_session=track_session,
            sleep_session=sleep_session,
            linearized=linearized,
            lin_row=lin_row,
            unit_table=unit_table,
            crossval=crossval,
            orientation_rule=orientation_rule,
            reversal_applied=reversal_applied,
            reasons=reasons,
            min_encoding_units=min_encoding_units,
        )
        figures = write_figures(
            animal=animal,
            date=date,
            track_session=track_session,
            output_dir=output_dir,
            unit_table=unit_table,
            place_fields=place_fields,
            crossval=crossval,
        )
        return row, unit_table, figures
    except Exception as exc:  # noqa: BLE001 - keep QC running across sessions.
        if not reasons:
            reasons.append(type(exc).__name__ + ":" + str(exc))
        row = failed_decoder_row(
            animal=animal,
            date=date,
            track_session=track_session,
            sleep_session=sleep_session,
            lin_row=lin_row,
            orientation_rule=orientation_rule,
            reversal_applied=reversal_applied,
            reasons=reasons,
            min_encoding_units=min_encoding_units,
        )
        return row, pd.DataFrame(columns=UNIT_COLUMNS), figures


def load_or_create_linearized_position(
    *,
    dataset_root: Path,
    linearization_root: Path,
    animal: str,
    date: str,
    track_session: str,
    output_dir: Path,
    position_bin_size_cm: float,
) -> pd.DataFrame:
    source = linearization_root / "sessions" / animal / date / "linearized_position.csv"
    if source.is_file():
        return pd.read_csv(source)
    target_dir = output_dir / "linearization"
    linearize_pos_file(
        _track_stem_path(dataset_root, animal, date, track_session).with_suffix(".pos"),
        target_dir,
        occupancy_bin_size_cm=position_bin_size_cm,
    )
    return pd.read_csv(target_dir / "linearized_position.csv")


def load_track_spikes(track_stem: Path, tetrodes: Sequence[int]) -> TrackSpikes:
    spike_times: list[float] = []
    unit_ids: list[int] = []
    units: set[int] = set()
    for tetrode in tetrodes:
        raw_path = track_stem.with_suffix(f".{int(tetrode)}")
        cut_path = track_stem.parent / f"{track_stem.name}_{int(tetrode)}.cut"
        if not raw_path.is_file() or not cut_path.is_file():
            continue
        cut = read_axona_cut(cut_path, tetrode_path=raw_path)
        if cut.spike_times_s is None:
            continue
        labels = np.asarray(cut.labels, dtype=int)
        times = np.asarray(cut.spike_times_s, dtype=float)
        keep = labels > 0
        ids = np.asarray([int(tetrode) * 100 + int(label) for label in labels[keep]], dtype=int)
        spike_times.extend(times[keep].tolist())
        unit_ids.extend(ids.tolist())
        units.update(ids.tolist())
    if not spike_times:
        return TrackSpikes(spike_times_s=np.empty(0, dtype=float), unit_ids=np.empty(0, dtype=int), units=())
    order = np.argsort(np.asarray(spike_times, dtype=float))
    return TrackSpikes(
        spike_times_s=np.asarray(spike_times, dtype=float)[order],
        unit_ids=np.asarray(unit_ids, dtype=int)[order],
        units=tuple(sorted(units)),
    )


def unit_qc_table(
    *,
    animal: str,
    date: str,
    track_session: str,
    linearized: pd.DataFrame,
    spikes: TrackSpikes,
    position_bin_size_cm: float,
    min_unit_spikes: int,
    min_unit_mean_rate_hz: float,
    min_place_information_bits: float,
    min_place_peak_rate_hz: float,
    smoothing_bins: int,
) -> tuple[pd.DataFrame, PlaceFields]:
    valid = valid_position_mask(linearized)
    times = pd.to_numeric(linearized["time_s"], errors="coerce").to_numpy(dtype=float)
    linear = pd.to_numeric(linearized["linear_position_cm"], errors="coerce").to_numpy(dtype=float)
    edges = position_edges(linear[valid], position_bin_size_cm)
    centers = 0.5 * (edges[:-1] + edges[1:])
    occupancy = occupancy_seconds(linear, times, valid, edges)
    fields = fit_place_fields(
        linear=linear,
        times=times,
        valid=valid,
        spikes=spikes,
        unit_ids=spikes.units,
        edges=edges,
        smoothing_bins=smoothing_bins,
    )
    duration = _duration(times[valid])
    occupancy_covered_fraction = float((occupancy > 0.0).sum() / len(occupancy)) if len(occupancy) else 0.0
    rows: list[dict[str, object]] = []
    for index, unit_id in enumerate(spikes.units):
        unit_spikes = spikes.spike_times_s[spikes.unit_ids == int(unit_id)]
        n_spikes = int(unit_spikes.shape[0])
        mean_rate = float(n_spikes / duration) if duration > 0 else np.nan
        rates = fields[index] if fields.size else np.full(centers.shape, np.nan)
        peak_rate = float(np.nanmax(rates)) if rates.size and np.isfinite(rates).any() else np.nan
        info = spatial_information(rates, occupancy)
        peak_cm = float(centers[int(np.nanargmax(rates))]) if rates.size and np.isfinite(rates).any() else np.nan
        width = place_field_width_cm(rates, centers, position_bin_size_cm)
        unit_pass = (
            n_spikes >= int(min_unit_spikes)
            and np.isfinite(mean_rate)
            and mean_rate >= float(min_unit_mean_rate_hz)
            and np.isfinite(peak_rate)
            and peak_rate > 0.0
        )
        rows.append(
            {
                "animal": animal,
                "date": date,
                "track1_session": track_session,
                "unit_id": int(unit_id),
                "n_spikes_track1": n_spikes,
                "mean_rate_hz": mean_rate,
                "peak_rate_hz": peak_rate,
                "spatial_information": info,
                "place_field_peak_cm": peak_cm,
                "place_field_width_cm": width,
                "occupancy_covered_fraction": occupancy_covered_fraction,
                "unit_qc_passed": bool(unit_pass),
            }
        )
    table = pd.DataFrame(rows, columns=UNIT_COLUMNS)
    place_like = (
        table["unit_qc_passed"].map(_as_bool)
        & (pd.to_numeric(table["spatial_information"], errors="coerce") >= float(min_place_information_bits))
        & (pd.to_numeric(table["peak_rate_hz"], errors="coerce") >= float(min_place_peak_rate_hz))
    )
    table.loc[:, "unit_qc_passed"] = table["unit_qc_passed"].map(_as_bool)
    table.attrs["n_place_like_units"] = int(place_like.sum())
    return table, PlaceFields(tuple(spikes.units), centers, occupancy, fields, occupancy_covered_fraction)


def crossval_decode(
    *,
    linearized: pd.DataFrame,
    spikes: TrackSpikes,
    unit_ids: tuple[int, ...],
    crossval_folds: int,
    position_bin_size_cm: float,
    decode_window_s: float,
    smoothing_bins: int,
) -> dict[str, object]:
    windows = decode_windows(linearized, decode_window_s)
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
    true_pos = pd.to_numeric(windows["true_position_cm"], errors="coerce").to_numpy(dtype=float)
    starts = pd.to_numeric(windows["start_time_s"], errors="coerce").to_numpy(dtype=float)
    ends = pd.to_numeric(windows["end_time_s"], errors="coerce").to_numpy(dtype=float)
    valid_window = np.isfinite(true_pos) & np.isfinite(starts) & np.isfinite(ends) & (ends > starts)
    if valid_window.sum() < max(2, int(crossval_folds)):
        return empty
    indices = np.flatnonzero(valid_window)
    folds = np.array_split(indices, min(int(crossval_folds), indices.shape[0]))
    posterior_predictions = np.full(windows.shape[0], np.nan, dtype=float)
    map_predictions = np.full(windows.shape[0], np.nan, dtype=float)
    linear = pd.to_numeric(linearized["linear_position_cm"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(linearized["time_s"], errors="coerce").to_numpy(dtype=float)
    valid_position = valid_position_mask(linearized)
    edges = position_edges(linear[valid_position], position_bin_size_cm)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for fold in folds:
        if fold.size == 0:
            continue
        test_mask = np.zeros(windows.shape[0], dtype=bool)
        test_mask[fold] = True
        train_intervals = [(float(starts[i]), float(ends[i])) for i in indices if not test_mask[i]]
        train_sample_mask = sample_mask_in_intervals(times, train_intervals) & valid_position
        rates = fit_place_fields(
            linear=linear,
            times=times,
            valid=train_sample_mask,
            spikes=spikes,
            unit_ids=unit_ids,
            edges=edges,
            smoothing_bins=smoothing_bins,
        )
        prior = occupancy_seconds(linear, times, train_sample_mask, edges)
        prior = (prior + 1e-6) / float(np.nansum(prior + 1e-6))
        for row_index in fold:
            counts = spike_counts_for_window(spikes, unit_ids, float(starts[row_index]), float(ends[row_index]))
            posterior = poisson_posterior(counts, rates, float(ends[row_index] - starts[row_index]), prior)
            if posterior.size and np.isfinite(posterior).all() and posterior.sum() > 0.0:
                posterior_predictions[row_index] = float(np.sum(posterior * centers))
                map_predictions[row_index] = float(centers[int(np.argmax(posterior))])
    decoded = valid_window & np.isfinite(posterior_predictions) & np.isfinite(map_predictions)
    if not np.any(decoded):
        return empty
    posterior_error = np.abs(posterior_predictions[decoded] - true_pos[decoded])
    map_error = np.abs(map_predictions[decoded] - true_pos[decoded])
    return {
        "crossval_n_folds": int(len(folds)),
        "posterior_mean_error_cm_median": percentile(posterior_error, 50.0),
        "posterior_mean_error_cm_p75": percentile(posterior_error, 75.0),
        "posterior_mean_error_cm_p90": percentile(posterior_error, 90.0),
        "map_error_cm_median": percentile(map_error, 50.0),
        "map_error_cm_p75": percentile(map_error, 75.0),
        "map_error_cm_p90": percentile(map_error, 90.0),
        "posterior_coverage_fraction": float(np.count_nonzero(decoded) / np.count_nonzero(valid_window)),
        "true_position_cm": true_pos[decoded],
        "posterior_mean_position_cm": posterior_predictions[decoded],
        "map_position_cm": map_predictions[decoded],
    }


def decoder_row(
    *,
    animal: str,
    date: str,
    track_session: str,
    sleep_session: str,
    linearized: pd.DataFrame,
    lin_row: pd.Series,
    unit_table: pd.DataFrame,
    crossval: dict[str, object],
    orientation_rule: str,
    reversal_applied: bool,
    reasons: list[str],
    min_encoding_units: int,
) -> dict[str, object]:
    unit_rates = pd.to_numeric(unit_table["mean_rate_hz"], errors="coerce") if not unit_table.empty else pd.Series(dtype=float)
    status = "pass" if not reasons else "fail"
    row = {
        "animal": animal,
        "date": date,
        "track1_session": track_session,
        "sleeppost_session": sleep_session,
        "n_units_track1": int(len(unit_table)),
        "n_position_samples": int(len(linearized)),
        "track_duration_s": _position_duration(linearized),
        "valid_position_fraction": float(lin_row["valid_position_fraction"]),
        "linearized_track_span_cm": float(lin_row["linearized_position_span_cm"]),
        "occupancy_nonzero_bins": int(lin_row["occupancy_nonzero_bins"]),
        "encoding_units_passing_qc": int(unit_table["unit_qc_passed"].map(_as_bool).sum()) if not unit_table.empty else 0,
        "median_firing_rate_hz": float(unit_rates.median()) if not unit_rates.dropna().empty else np.nan,
        "n_place_like_units": int(unit_table.attrs.get("n_place_like_units", 0)),
        "crossval_n_folds": int(crossval["crossval_n_folds"]),
        "posterior_mean_error_cm_median": float(crossval["posterior_mean_error_cm_median"]),
        "posterior_mean_error_cm_p75": float(crossval["posterior_mean_error_cm_p75"]),
        "posterior_mean_error_cm_p90": float(crossval["posterior_mean_error_cm_p90"]),
        "map_error_cm_median": float(crossval["map_error_cm_median"]),
        "map_error_cm_p75": float(crossval["map_error_cm_p75"]),
        "map_error_cm_p90": float(crossval["map_error_cm_p90"]),
        "posterior_coverage_fraction": float(crossval["posterior_coverage_fraction"]),
        "decoder_status": status,
        "orientation_rule": orientation_rule,
        "reversal_applied": bool(reversal_applied),
        "exclusion_reason": ";".join(reasons),
    }
    scoring_reasons = decoder_scoring_available_reasons(row, min_encoding_units=min_encoding_units)
    row["decoder_qc_paper_ready"] = bool(status == "pass")
    row["decoder_qc_scoring_available"] = bool(not scoring_reasons)
    row["decoder_scoring_available_reason"] = ";".join(scoring_reasons)
    return row


def failed_decoder_row(
    *,
    animal: str,
    date: str,
    track_session: str,
    sleep_session: str,
    lin_row: pd.Series | None,
    orientation_rule: str,
    reversal_applied: bool,
    reasons: list[str],
    min_encoding_units: int,
) -> dict[str, object]:
    row = {column: np.nan for column in DECODER_COLUMNS}
    row.update(
        {
            "animal": animal,
            "date": date,
            "track1_session": track_session,
            "sleeppost_session": sleep_session,
            "n_units_track1": 0,
            "n_position_samples": 0,
            "track_duration_s": np.nan,
            "valid_position_fraction": float(lin_row["valid_position_fraction"]) if lin_row is not None else np.nan,
            "linearized_track_span_cm": float(lin_row["linearized_position_span_cm"]) if lin_row is not None else np.nan,
            "occupancy_nonzero_bins": int(lin_row["occupancy_nonzero_bins"]) if lin_row is not None else 0,
            "encoding_units_passing_qc": 0,
            "n_place_like_units": 0,
            "crossval_n_folds": 0,
            "posterior_coverage_fraction": 0.0,
            "decoder_status": "fail",
            "decoder_qc_paper_ready": False,
            "orientation_rule": orientation_rule,
            "reversal_applied": bool(reversal_applied),
            "exclusion_reason": ";".join(reasons),
        }
    )
    scoring_reasons = decoder_scoring_available_reasons(row, min_encoding_units=min_encoding_units)
    row["decoder_qc_scoring_available"] = bool(not scoring_reasons)
    row["decoder_scoring_available_reason"] = ";".join(scoring_reasons)
    return row


def decoder_scoring_available_reasons(row: dict[str, object], *, min_encoding_units: int) -> list[str]:
    """Return non-paper reasons that still block a technical scoring smoke."""

    reasons: list[str] = []
    n_position = _numeric_value(row.get("n_position_samples"))
    valid_fraction = _numeric_value(row.get("valid_position_fraction"))
    track_span = _numeric_value(row.get("linearized_track_span_cm"))
    occupancy_bins = _numeric_value(row.get("occupancy_nonzero_bins"))
    units = _numeric_value(row.get("encoding_units_passing_qc"))
    posterior_error = _numeric_value(row.get("posterior_mean_error_cm_median"))
    map_error = _numeric_value(row.get("map_error_cm_median"))
    coverage = _numeric_value(row.get("posterior_coverage_fraction"))

    if not np.isfinite(n_position) or n_position <= 0:
        reasons.append("missing_track1_position_samples")
    if not np.isfinite(valid_fraction) or valid_fraction <= 0.0:
        reasons.append("invalid_track1_position_fraction")
    if not np.isfinite(track_span) or track_span <= 0.0:
        reasons.append("invalid_linearized_track_span")
    if not np.isfinite(occupancy_bins) or occupancy_bins <= 0:
        reasons.append("missing_linearized_occupancy")
    if not np.isfinite(units) or units < int(min_encoding_units):
        reasons.append("too_few_encoding_units")
    if not np.isfinite(posterior_error):
        reasons.append("missing_posterior_mean_error")
    if not np.isfinite(map_error):
        reasons.append("missing_map_error")
    if not np.isfinite(coverage) or coverage <= 0.0:
        reasons.append("missing_posterior_coverage")
    if np.isfinite(posterior_error) and posterior_error < 0.0:
        reasons.append("invalid_posterior_mean_error")
    if np.isfinite(map_error) and map_error < 0.0:
        reasons.append("invalid_map_error")
    if np.isfinite(track_span) and track_span > 0.0:
        if np.isfinite(posterior_error) and posterior_error > track_span:
            reasons.append("posterior_mean_error_exceeds_track_span")
        if np.isfinite(map_error) and map_error > track_span:
            reasons.append("map_error_exceeds_track_span")
    if str(row.get("animal", "")).upper() == "R2142" and not _as_bool(row.get("reversal_applied", False)):
        reasons.append("r2142_reversal_not_applied")
    return reasons


def _numeric_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_by_animal(decoder: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "animal",
        "track1_sessions",
        "decoder_pass_sessions",
        "decoder_scoring_available_sessions",
        "total_encoding_units_passing_qc",
        "median_posterior_mean_error_cm",
        "median_map_error_cm",
        "animal_retained_after_decoder_qc",
        "animal_retained_for_decoder_scoring",
    ]
    if decoder.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for animal, group in decoder.groupby("animal", sort=True):
        passed = group[group["decoder_status"].astype(str).eq("pass")]
        scoring_available = group[group["decoder_qc_scoring_available"].map(_as_bool)]
        animal_units = units[units["animal"].astype(str).eq(str(animal))] if not units.empty else pd.DataFrame(columns=UNIT_COLUMNS)
        rows.append(
            {
                "animal": animal,
                "track1_sessions": int(len(group)),
                "decoder_pass_sessions": int(len(passed)),
                "decoder_scoring_available_sessions": int(len(scoring_available)),
                "total_encoding_units_passing_qc": int(animal_units["unit_qc_passed"].map(_as_bool).sum()) if not animal_units.empty else 0,
                "median_posterior_mean_error_cm": _median(group, "posterior_mean_error_cm_median"),
                "median_map_error_cm": _median(group, "map_error_cm_median"),
                "animal_retained_after_decoder_qc": bool(len(passed) > 0),
                "animal_retained_for_decoder_scoring": bool(len(scoring_available) > 0),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def gate_summary(
    *,
    pairs: pd.DataFrame,
    decoder: pd.DataFrame,
    min_encoding_units: int,
    max_posterior_median_error_cm: float,
    max_map_median_error_cm: float,
    min_posterior_coverage_fraction: float,
    max_decoder_animal_fraction: float,
    max_decoder_session_fraction: float,
) -> pd.DataFrame:
    expected_sessions = int(len(pairs))
    passed = decoder[decoder["decoder_status"].astype(str).eq("pass")] if not decoder.empty else decoder
    scoring_available = decoder[decoder["decoder_qc_scoring_available"].map(_as_bool)] if not decoder.empty else decoder
    input_animals = set(pairs["animal"].astype(str).str.upper()) if not pairs.empty else set()
    retained_animals = set(passed["animal"].astype(str).str.upper()) if not passed.empty else set()
    scoring_animals = set(scoring_available["animal"].astype(str).str.upper()) if not scoring_available.empty else set()
    r2142_rows = decoder[decoder["animal"].astype(str).str.upper().eq("R2142")] if not decoder.empty else decoder
    gates = [
        _gate(
            "track1_decoder_outputs_present",
            expected_sessions > 0 and len(decoder) == expected_sessions,
            f"sessions={len(decoder)}; usable_pairs={expected_sessions}",
            "all usable Track1 sessions produce decoder QC rows",
            "Decoder QC should cover the full usable pair table.",
        ),
        _gate(
            "animals_retained_after_decoder_qc",
            bool(input_animals) and input_animals == retained_animals,
            f"input_animals={len(input_animals)}; retained_animals={len(retained_animals)}",
            "all animals have at least one passing decoder QC row",
            "Avoids carrying a decoder into replay evidence for only one animal.",
        ),
        _gate(
            "animals_retained_for_decoder_scoring",
            bool(input_animals) and input_animals == scoring_animals,
            f"input_animals={len(input_animals)}; scoring_animals={len(scoring_animals)}",
            "all animals have at least one scoring-available decoder row",
            "This debug tier is for technical scoring smoke only, not paper-ready decoder claims.",
        ),
        _gate(
            "encoding_units_per_session",
            expected_sessions > 0 and _all_numeric_at_least(decoder, "encoding_units_passing_qc", min_encoding_units),
            f"passing={_count_numeric_at_least(decoder, 'encoding_units_passing_qc', min_encoding_units)}/{expected_sessions}",
            f"encoding_units_passing_qc >= {int(min_encoding_units)} for every retained session",
            "Every retained Track1 session should have enough sorted units for a decoder.",
        ),
        _gate(
            "finite_crossval_errors",
            expected_sessions > 0
            and _all_numeric_finite(decoder, "posterior_mean_error_cm_median")
            and _all_numeric_finite(decoder, "map_error_cm_median"),
            (
                f"posterior_finite={_count_numeric_finite(decoder, 'posterior_mean_error_cm_median')}/{expected_sessions}; "
                f"map_finite={_count_numeric_finite(decoder, 'map_error_cm_median')}/{expected_sessions}"
            ),
            "all retained sessions have finite cross-validated posterior/MAP errors",
            "A replay evidence failure should not be confused with an unscorable encoding model.",
        ),
        _gate(
            "posterior_median_error_below_threshold",
            expected_sessions > 0 and _all_numeric_at_most(decoder, "posterior_mean_error_cm_median", max_posterior_median_error_cm),
            f"passing={_count_numeric_at_most(decoder, 'posterior_mean_error_cm_median', max_posterior_median_error_cm)}/{expected_sessions}",
            f"posterior_mean_error_cm_median <= {float(max_posterior_median_error_cm):g}",
            "Predeclared decoder-quality threshold for posterior-mean decoding.",
        ),
        _gate(
            "map_median_error_below_threshold",
            expected_sessions > 0 and _all_numeric_at_most(decoder, "map_error_cm_median", max_map_median_error_cm),
            f"passing={_count_numeric_at_most(decoder, 'map_error_cm_median', max_map_median_error_cm)}/{expected_sessions}",
            f"map_error_cm_median <= {float(max_map_median_error_cm):g}",
            "Predeclared decoder-quality threshold for MAP decoding.",
        ),
        _gate(
            "posterior_coverage_fraction_acceptable",
            expected_sessions > 0 and _all_numeric_at_least(decoder, "posterior_coverage_fraction", min_posterior_coverage_fraction),
            f"passing={_count_numeric_at_least(decoder, 'posterior_coverage_fraction', min_posterior_coverage_fraction)}/{expected_sessions}",
            f"posterior_coverage_fraction >= {float(min_posterior_coverage_fraction):g}",
            "Most cross-validation windows should produce a finite posterior.",
        ),
        _gate(
            "decoder_coverage_not_animal_dominated",
            len(passed) > 0 and _max_group_fraction(passed, "animal") <= float(max_decoder_animal_fraction),
            f"max_animal_fraction={_max_group_fraction(passed, 'animal'):.6g}",
            f"max passing-session animal fraction <= {float(max_decoder_animal_fraction):g}",
            "Avoids a decoder-ready set dominated by one animal.",
        ),
        _gate(
            "decoder_coverage_not_session_dominated",
            len(passed) > 0 and _max_group_fraction(passed, "track1_session") <= float(max_decoder_session_fraction),
            f"max_session_fraction={_max_group_fraction(passed, 'track1_session'):.6g}",
            f"max passing-session session fraction <= {float(max_decoder_session_fraction):g}",
            "Avoids a decoder-ready set dominated by one session.",
        ),
        _gate(
            "r2142_reversal_explicit",
            r2142_rows.empty or bool(r2142_rows["reversal_applied"].map(_as_bool).all()),
            f"r2142_rows={len(r2142_rows)}; reversal_applied={bool(r2142_rows['reversal_applied'].map(_as_bool).all()) if not r2142_rows.empty else 'not_present'}",
            "R2142 rows explicitly carry reversal_applied=true",
            "The known R2142 tetrode reversal remains visible through decoder QC.",
        ),
    ]
    return pd.DataFrame(gates)


def write_figures(
    *,
    animal: str,
    date: str,
    track_session: str,
    output_dir: Path,
    unit_table: pd.DataFrame,
    place_fields: PlaceFields,
    crossval: dict[str, object],
) -> list[dict[str, object]]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        write_place_field_figure(figures_dir, animal, date, track_session, unit_table, place_fields),
        write_error_histogram(figures_dir, animal, date, track_session, crossval),
        write_predicted_vs_true_figure(figures_dir, animal, date, track_session, crossval),
    ]
    return rows


def write_place_field_figure(
    figures_dir: Path,
    animal: str,
    date: str,
    track_session: str,
    unit_table: pd.DataFrame,
    place_fields: PlaceFields,
) -> dict[str, object]:
    path = figures_dir / "encoding_place_fields.png"
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    if place_fields.rates_hz.size:
        order = unit_table.sort_values("place_field_peak_cm")["unit_id"].astype(int).tolist() if not unit_table.empty else list(place_fields.unit_ids)
        index = [place_fields.unit_ids.index(unit_id) for unit_id in order if unit_id in place_fields.unit_ids]
        image = place_fields.rates_hz[index, :] if index else place_fields.rates_hz
        ax.imshow(image, aspect="auto", origin="lower", extent=[place_fields.bin_centers_cm.min(), place_fields.bin_centers_cm.max(), 0, image.shape[0]])
    ax.set_xlabel("linear position (cm)")
    ax.set_ylabel("unit")
    ax.set_title(f"{animal} {date} Track1 place fields")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return figure_row(animal, date, track_session, "encoding_place_fields", path)


def write_error_histogram(figures_dir: Path, animal: str, date: str, track_session: str, crossval: dict[str, object]) -> dict[str, object]:
    path = figures_dir / "decoder_error_histogram.png"
    true_pos = np.asarray(crossval.get("true_position_cm", []), dtype=float)
    mean_pos = np.asarray(crossval.get("posterior_mean_position_cm", []), dtype=float)
    map_pos = np.asarray(crossval.get("map_position_cm", []), dtype=float)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    if true_pos.size:
        ax.hist(np.abs(mean_pos - true_pos), bins=30, alpha=0.6, label="posterior mean")
        ax.hist(np.abs(map_pos - true_pos), bins=30, alpha=0.6, label="MAP")
        ax.legend(loc="best")
    ax.set_xlabel("absolute decoding error (cm)")
    ax.set_ylabel("windows")
    ax.set_title(f"{animal} {date} decoder error")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return figure_row(animal, date, track_session, "decoder_error_histogram", path)


def write_predicted_vs_true_figure(figures_dir: Path, animal: str, date: str, track_session: str, crossval: dict[str, object]) -> dict[str, object]:
    path = figures_dir / "decoder_predicted_vs_true.png"
    true_pos = np.asarray(crossval.get("true_position_cm", []), dtype=float)
    mean_pos = np.asarray(crossval.get("posterior_mean_position_cm", []), dtype=float)
    map_pos = np.asarray(crossval.get("map_position_cm", []), dtype=float)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    if true_pos.size:
        ax.plot(true_pos, mean_pos, ".", ms=3, alpha=0.5, label="posterior mean")
        ax.plot(true_pos, map_pos, ".", ms=3, alpha=0.5, label="MAP")
        lo = float(np.nanmin(true_pos))
        hi = float(np.nanmax(true_pos))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.legend(loc="best")
    ax.set_xlabel("true position (cm)")
    ax.set_ylabel("decoded position (cm)")
    ax.set_title(f"{animal} {date}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return figure_row(animal, date, track_session, "decoder_predicted_vs_true", path)


def build_markdown_summary(
    decoder: pd.DataFrame,
    animals: pd.DataFrame,
    gates: pd.DataFrame,
    *,
    min_encoding_units: int,
    max_posterior_median_error_cm: float,
    max_map_median_error_cm: float,
    min_posterior_coverage_fraction: float,
    crossval_folds: int,
    position_bin_size_cm: float,
    decode_window_s: float,
) -> str:
    pass_sessions = int(decoder["decoder_status"].astype(str).eq("pass").sum()) if not decoder.empty else 0
    scoring_sessions = int(decoder["decoder_qc_scoring_available"].map(_as_bool).sum()) if not decoder.empty else 0
    total_sessions = int(len(decoder))
    pass_animals = int(animals["animal_retained_after_decoder_qc"].map(_as_bool).sum()) if not animals.empty else 0
    scoring_animals = int(animals["animal_retained_for_decoder_scoring"].map(_as_bool).sum()) if not animals.empty else 0
    gate_passes = int(gates["passed"].map(_as_bool).sum()) if not gates.empty else 0
    lines = [
        "# Olafsdottir Track1 Decoder QC Summary",
        "",
        "This is an encoding/decoder readiness checkpoint only. It does not score SleepPOST replay evidence or compare 1D against 2D.",
        "",
        "## Defaults",
        "",
        _markdown_table(
            ["Parameter", "Value"],
            [
                ("minimum encoding units per session", min_encoding_units),
                ("posterior median error threshold cm", max_posterior_median_error_cm),
                ("MAP median error threshold cm", max_map_median_error_cm),
                ("posterior coverage threshold", min_posterior_coverage_fraction),
                ("cross-validation folds", crossval_folds),
                ("position bin size cm", position_bin_size_cm),
                ("decode window s", decode_window_s),
            ],
        ),
        "",
        "## Overview",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ("Track1 sessions processed", total_sessions),
                ("Track1 sessions passing decoder QC", pass_sessions),
                ("Track1 sessions scoring-available", scoring_sessions),
                ("Animals processed", len(animals)),
                ("Animals retained after decoder QC", pass_animals),
                ("Animals retained for decoder scoring", scoring_animals),
                ("Readiness gates passed", f"{gate_passes}/{len(gates)}"),
            ],
        ),
        "",
        "## Gate Summary",
        "",
        _markdown_table(["Gate", "Status", "Value"], gates[["gate", "status", "value"]].itertuples(index=False, name=None)),
        "",
        "## Animal Summary",
        "",
        _markdown_table(["Animal", "Sessions", "Paper-ready", "Scoring-available", "Median posterior error", "Median MAP error"], animal_summary_rows(animals)),
        "",
    ]
    return "\n".join(lines)


def fit_place_fields(
    *,
    linear: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    spikes: TrackSpikes,
    unit_ids: Sequence[int],
    edges: np.ndarray,
    smoothing_bins: int,
) -> np.ndarray:
    occupancy = occupancy_seconds(linear, times, valid, edges)
    fields = np.zeros((len(unit_ids), edges.shape[0] - 1), dtype=float)
    if len(unit_ids) == 0:
        return fields
    for unit_index, unit_id in enumerate(unit_ids):
        unit_times = spikes.spike_times_s[spikes.unit_ids == int(unit_id)]
        spike_pos = interpolate_position_at_times(unit_times, times, linear, valid)
        counts, _ = np.histogram(spike_pos[np.isfinite(spike_pos)], bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = counts / occupancy
        rates[~np.isfinite(rates)] = 0.0
        fields[unit_index, :] = smooth_1d(rates, smoothing_bins)
    return fields


def occupancy_seconds(linear: np.ndarray, times: np.ndarray, valid: np.ndarray, edges: np.ndarray) -> np.ndarray:
    dt = sample_durations(times)
    keep = np.asarray(valid, dtype=bool) & np.isfinite(linear) & np.isfinite(dt)
    bins = np.searchsorted(edges, linear[keep], side="right") - 1
    bins = np.clip(bins, 0, edges.shape[0] - 2)
    occupancy = np.zeros(edges.shape[0] - 1, dtype=float)
    np.add.at(occupancy, bins, dt[keep])
    return occupancy


def decode_windows(linearized: pd.DataFrame, decode_window_s: float) -> pd.DataFrame:
    times = pd.to_numeric(linearized["time_s"], errors="coerce").to_numpy(dtype=float)
    linear = pd.to_numeric(linearized["linear_position_cm"], errors="coerce").to_numpy(dtype=float)
    valid = valid_position_mask(linearized)
    if not np.any(valid):
        return pd.DataFrame(columns=["start_time_s", "end_time_s", "true_position_cm"])
    start = float(np.nanmin(times[valid]))
    end = float(np.nanmax(times[valid]))
    edges = np.arange(start, end + float(decode_window_s), float(decode_window_s))
    rows: list[dict[str, float]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        keep = valid & (times >= left) & (times < right)
        if np.any(keep):
            rows.append(
                {
                    "start_time_s": float(left),
                    "end_time_s": float(right),
                    "true_position_cm": float(np.nanmedian(linear[keep])),
                }
            )
    return pd.DataFrame(rows)


def poisson_posterior(counts: np.ndarray, rates: np.ndarray, duration_s: float, prior: np.ndarray) -> np.ndarray:
    if rates.size == 0:
        return np.asarray([], dtype=float)
    safe_rates = np.maximum(np.asarray(rates, dtype=float), 1e-6)
    counts = np.asarray(counts, dtype=float)
    logp = np.log(np.maximum(prior, 1e-12))
    logp = logp + counts @ np.log(safe_rates * float(duration_s))
    logp = logp - float(duration_s) * np.sum(safe_rates, axis=0)
    logp = logp - float(np.nanmax(logp))
    posterior = np.exp(logp)
    total = float(np.sum(posterior))
    return posterior / total if total > 0.0 and np.isfinite(total) else np.full(prior.shape, np.nan)


def spike_counts_for_window(spikes: TrackSpikes, unit_ids: Sequence[int], start_s: float, end_s: float) -> np.ndarray:
    counts = np.zeros(len(unit_ids), dtype=float)
    in_window = (spikes.spike_times_s >= float(start_s)) & (spikes.spike_times_s < float(end_s))
    if not np.any(in_window):
        return counts
    ids = spikes.unit_ids[in_window]
    for index, unit_id in enumerate(unit_ids):
        counts[index] = float(np.count_nonzero(ids == int(unit_id)))
    return counts


def interpolate_position_at_times(spike_times: np.ndarray, times: np.ndarray, linear: np.ndarray, valid: np.ndarray) -> np.ndarray:
    keep = np.asarray(valid, dtype=bool) & np.isfinite(times) & np.isfinite(linear)
    if np.count_nonzero(keep) < 2 or spike_times.size == 0:
        return np.full(spike_times.shape, np.nan, dtype=float)
    values = np.interp(spike_times, times[keep], linear[keep], left=np.nan, right=np.nan)
    return values


def position_edges(values: np.ndarray, bin_size_cm: float) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([0.0, float(bin_size_cm)], dtype=float)
    lo = 0.0
    hi = max(float(np.nanmax(finite)), float(bin_size_cm))
    return np.arange(lo, hi + float(bin_size_cm), float(bin_size_cm))


def sample_durations(times: np.ndarray) -> np.ndarray:
    arr = np.asarray(times, dtype=float)
    if arr.size == 0:
        return arr
    if arr.size == 1:
        return np.asarray([0.0], dtype=float)
    dt = np.diff(arr, append=arr[-1] + np.nanmedian(np.diff(arr)))
    dt[~np.isfinite(dt) | (dt <= 0.0)] = np.nanmedian(dt[np.isfinite(dt) & (dt > 0.0)])
    return dt


def spatial_information(rates: np.ndarray, occupancy: np.ndarray) -> float:
    occ = np.asarray(occupancy, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if occ.size == 0 or rates.size == 0 or np.nansum(occ) <= 0.0:
        return np.nan
    p = occ / float(np.nansum(occ))
    mean_rate = float(np.nansum(p * rates))
    if mean_rate <= 0.0:
        return 0.0
    ratio = rates / mean_rate
    keep = (p > 0.0) & (ratio > 0.0) & np.isfinite(ratio)
    return float(np.nansum(p[keep] * ratio[keep] * np.log2(ratio[keep])))


def place_field_width_cm(rates: np.ndarray, centers: np.ndarray, bin_size_cm: float) -> float:
    arr = np.asarray(rates, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any() or np.nanmax(arr) <= 0.0:
        return np.nan
    peak = int(np.nanargmax(arr))
    threshold = 0.5 * float(arr[peak])
    left = peak
    while left > 0 and arr[left - 1] >= threshold:
        left -= 1
    right = peak
    while right < arr.size - 1 and arr[right + 1] >= threshold:
        right += 1
    return float((right - left + 1) * float(bin_size_cm))


def sample_mask_in_intervals(times: np.ndarray, intervals: Sequence[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in intervals:
        mask |= (times >= float(start)) & (times < float(end))
    return mask


def valid_position_mask(linearized: pd.DataFrame) -> np.ndarray:
    valid = linearized["valid_position"].map(_as_bool).to_numpy(dtype=bool) if "valid_position" in linearized else np.zeros(len(linearized), dtype=bool)
    linear = pd.to_numeric(linearized.get("linear_position_cm", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(linearized.get("time_s", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    return valid & np.isfinite(linear) & np.isfinite(times)


def smooth_1d(values: np.ndarray, smoothing_bins: int) -> np.ndarray:
    window = max(int(smoothing_bins), 1)
    if window <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def matching_linearization_row(linearization: pd.DataFrame, animal: str, date: str, track_session: str) -> pd.Series | None:
    rows = linearization[
        linearization["animal"].astype(str).str.upper().eq(str(animal).upper())
        & linearization["date"].astype(str).eq(str(date))
        & linearization["track_session"].astype(str).eq(str(track_session))
    ]
    return rows.iloc[0] if not rows.empty else None


def _track_stem_path(dataset_root: Path, animal: str, date: str, track_session: str) -> Path:
    return dataset_root / animal.lower() / date / track_session


def _parse_tetrodes(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(raw).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            return ()
    return tuple(values)


def _position_duration(linearized: pd.DataFrame) -> float:
    if "time_s" not in linearized or linearized.empty:
        return np.nan
    times = pd.to_numeric(linearized["time_s"], errors="coerce").dropna()
    return float(times.max() - times.min()) if not times.empty else np.nan


def _duration(times: np.ndarray) -> float:
    finite = np.asarray(times, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.max() - finite.min()) if finite.size >= 2 else 0.0


def percentile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if finite.size else np.nan


def figure_row(animal: str, date: str, track_session: str, figure_type: str, path: Path) -> dict[str, object]:
    return {
        "animal": animal,
        "date": date,
        "track1_session": track_session,
        "figure_type": figure_type,
        "figure_path": str(path),
    }


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _all_numeric_finite(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and np.isfinite(values).all())


def _count_numeric_finite(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce")
    return int((values.notna() & np.isfinite(values)).sum())


def _all_numeric_at_least(frame: pd.DataFrame, column: str, threshold: float) -> bool:
    if frame.empty or column not in frame:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and (values >= float(threshold)).all())


def _count_numeric_at_least(frame: pd.DataFrame, column: str, threshold: float) -> int:
    if frame.empty or column not in frame:
        return 0
    return int((pd.to_numeric(frame[column], errors="coerce").fillna(-np.inf) >= float(threshold)).sum())


def _all_numeric_at_most(frame: pd.DataFrame, column: str, threshold: float) -> bool:
    if frame.empty or column not in frame:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and (values <= float(threshold)).all())


def _count_numeric_at_most(frame: pd.DataFrame, column: str, threshold: float) -> int:
    if frame.empty or column not in frame:
        return 0
    return int((pd.to_numeric(frame[column], errors="coerce").fillna(np.inf) <= float(threshold)).sum())


def _max_group_fraction(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    counts = frame.groupby(column).size()
    return float(counts.max() / counts.sum()) if counts.sum() else np.nan


def _gate(gate: str, passed: bool, value: str, requirement: str, note: str) -> dict[str, object]:
    return {
        "gate": gate,
        "passed": bool(passed),
        "status": "pass" if passed else "fail",
        "value": value,
        "requirement": requirement,
        "note": note,
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def animal_summary_rows(animals: pd.DataFrame) -> list[tuple[object, ...]]:
    if animals.empty:
        return []
    rows: list[tuple[object, ...]] = []
    for row in animals.itertuples(index=False):
        rows.append(
            (
                row.animal,
                int(row.track1_sessions),
                int(row.decoder_pass_sessions),
                int(row.decoder_scoring_available_sessions),
                f"{float(row.median_posterior_mean_error_cm):.3g}" if np.isfinite(float(row.median_posterior_mean_error_cm)) else "nan",
                f"{float(row.median_map_error_cm):.3g}" if np.isfinite(float(row.median_map_error_cm)) else "nan",
            )
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--linearization-qc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/olafsdottir-track1-decoder-qc"))
    parser.add_argument("--min-encoding-units", type=int, default=5)
    parser.add_argument("--max-posterior-median-error-cm", type=float, default=35.0)
    parser.add_argument("--max-map-median-error-cm", type=float, default=45.0)
    parser.add_argument("--min-posterior-coverage-fraction", type=float, default=0.80)
    parser.add_argument("--crossval-folds", type=int, default=5)
    parser.add_argument("--position-bin-size-cm", type=float, default=5.0)
    parser.add_argument("--decode-window-s", type=float, default=0.250)
    parser.add_argument("--min-unit-spikes", type=int, default=5)
    parser.add_argument("--min-unit-mean-rate-hz", type=float, default=0.01)
    parser.add_argument("--min-place-information-bits", type=float, default=0.05)
    parser.add_argument("--min-place-peak-rate-hz", type=float, default=0.05)
    parser.add_argument("--smoothing-bins", type=int, default=1)
    parser.add_argument("--max-decoder-animal-fraction", type=float, default=0.75)
    parser.add_argument("--max-decoder-session-fraction", type=float, default=0.75)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tables = run_decoder_qc(
        dataset_root=args.dataset_root,
        pairs_csv=args.pairs_csv,
        linearization_qc=args.linearization_qc,
        output_dir=args.output_dir,
        min_encoding_units=args.min_encoding_units,
        max_posterior_median_error_cm=args.max_posterior_median_error_cm,
        max_map_median_error_cm=args.max_map_median_error_cm,
        min_posterior_coverage_fraction=args.min_posterior_coverage_fraction,
        crossval_folds=args.crossval_folds,
        position_bin_size_cm=args.position_bin_size_cm,
        decode_window_s=args.decode_window_s,
        min_unit_spikes=args.min_unit_spikes,
        min_unit_mean_rate_hz=args.min_unit_mean_rate_hz,
        min_place_information_bits=args.min_place_information_bits,
        min_place_peak_rate_hz=args.min_place_peak_rate_hz,
        smoothing_bins=args.smoothing_bins,
        max_decoder_animal_fraction=args.max_decoder_animal_fraction,
        max_decoder_session_fraction=args.max_decoder_session_fraction,
    )
    print(tables["decoder"].to_string(index=False))
    print()
    print(tables["gates"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
