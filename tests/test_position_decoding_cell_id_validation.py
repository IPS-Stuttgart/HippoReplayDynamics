from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.position_validation import _spike_counts_for_window, fit_place_field_encoding_for_position_mask


def _minimal_session(spikes: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="Open1",
        path=Path("."),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.asarray(spikes, dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 1.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _encoding() -> EncodingModel:
    config = EncodingConfig(min_speed_cm_s=0.0, smoothing_sigma_bins=0.0, use_excitatory=False)
    return EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 1), dtype=float),
        occupancy_s=np.ones(1, dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=config,
    )


def test_position_mask_encoding_rejects_fractional_spike_cell_ids() -> None:
    session = _minimal_session(np.array([[0.25, 1.5]], dtype=float))
    config = EncodingConfig(min_speed_cm_s=0.0, smoothing_sigma_bins=0.0, use_excitatory=False)

    with pytest.raises(ValueError, match="spike cell IDs.*integer-valued"):
        fit_place_field_encoding_for_position_mask(
            session,
            np.array([True, True, True], dtype=bool),
            config,
        )


def test_position_window_counts_reject_fractional_spike_cell_ids() -> None:
    session = _minimal_session(np.array([[0.25, 1.5]], dtype=float))

    with pytest.raises(ValueError, match="spike cell IDs.*integer-valued"):
        _spike_counts_for_window(session, _encoding(), 0.0, 1.0)
