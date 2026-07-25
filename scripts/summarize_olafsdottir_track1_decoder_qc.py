#!/usr/bin/env python3
"""Track1 decoder QC entry point with enforced place-field unit gates."""

from __future__ import annotations

import importlib

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


_impl.unit_qc_table = unit_qc_table


if __name__ == "__main__":
    raise SystemExit(_impl.main())
