from types import SimpleNamespace

import numpy as np
import pandas as pd

import hipporeplayimm.position_validation as validation

_SAMPLE_COLUMNS = (
    "session",
    "fold",
    "window_index",
    "start_time",
    "end_time",
    "center_time",
    "true_x",
    "true_y",
    "posterior_mean_x",
    "posterior_mean_y",
    "map_x",
    "map_y",
    "map_bin",
    "true_bin",
    "posterior_mean_error_cm",
    "map_error_cm",
    "true_bin_probability",
    "true_bin_rank",
    "posterior_entropy",
    "n_spikes",
    "n_cells",
    "n_position_bins",
    "observation_model",
    "spike_mark_features",
    "spike_mark_source",
    "clusterless_mark_likelihood",
)

_SUMMARY_COLUMNS = (
    "session",
    "decode_windows",
    "folds",
    "median_posterior_mean_error_cm",
    "median_map_error_cm",
    "mean_true_bin_probability",
    "median_true_bin_rank",
    "mean_spikes_per_window",
    "cells",
    "spatial_bins",
    "spike_mark_features",
    "clusterless_mark_likelihood",
)


def test_empty_session_position_decoding_keeps_sample_schema() -> None:
    session = SimpleNamespace(
        position=np.empty((0, 3), dtype=float),
        spikes=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        run_times=np.empty((0, 2), dtype=float),
    )

    samples = validation.validate_session_position_decoding(session)

    assert samples.empty
    assert tuple(samples.columns) == _SAMPLE_COLUMNS


def test_empty_position_decoding_run_writes_readable_csvs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(validation, "load_open_field_sessions", lambda _root: [])

    result = validation.run_position_decoding_validation("unused")

    assert result.samples.empty
    assert tuple(result.samples.columns) == _SAMPLE_COLUMNS
    assert result.summary.empty
    assert tuple(result.summary.columns) == _SUMMARY_COLUMNS

    result.write(tmp_path)

    for filename in ("position_decoding_samples.csv", "position_decoding_scores.csv"):
        frame = pd.read_csv(tmp_path / filename)
        assert frame.empty
        assert tuple(frame.columns) == _SAMPLE_COLUMNS

    summary = pd.read_csv(tmp_path / "position_decoding_summary.csv")
    assert summary.empty
    assert tuple(summary.columns) == _SUMMARY_COLUMNS
