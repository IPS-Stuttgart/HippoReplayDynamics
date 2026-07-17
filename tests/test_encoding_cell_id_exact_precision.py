from __future__ import annotations

from pathlib import Path

import numpy as np

import hipporeplayimm.encoding as encoding_module
import hipporeplayimm.position_validation as position_validation
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig


def _large_id_session() -> tuple[ReplaySession, int, int]:
    first = 2**53
    second = first + 1
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("."),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 10.0, 1.0],
                [2.0, 20.0, 2.0],
            ],
            dtype=float,
        ),
        spikes=np.array(
            [
                [0.5, first],
                [1.5, second],
            ],
            dtype=object,
        ),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([first, second], dtype=object),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )
    return session, first, second


def _encoding_config() -> EncodingConfig:
    return EncodingConfig(
        bin_size_cm=5.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=0.0,
        min_occupancy_s=1e-3,
        rate_floor_hz=1e-4,
        arena_padding_cm=1.0,
        use_excitatory=True,
        exclude_ripple_intervals=False,
    )


def _assert_both_cells_received_spikes(model: encoding_module.EncodingModel) -> None:
    above_floor = model.rates_hz > model.config.rate_floor_hz
    assert above_floor.any(axis=1).tolist() == [True, True]


def test_encoding_spike_selection_preserves_exact_large_ids() -> None:
    session, first, second = _large_id_session()

    spikes, cell_ids = encoding_module._spikes_and_cell_ids_for_encoding(
        session,
        _encoding_config(),
    )

    assert spikes[:, 0].dtype == np.dtype(float)
    assert spikes[:, 1].dtype == np.dtype(int)
    assert spikes[:, 1].tolist() == [first, second]
    assert cell_ids.tolist() == [first, second]


def test_place_field_encoding_keeps_adjacent_large_ids_distinct() -> None:
    session, first, second = _large_id_session()

    model = encoding_module.fit_place_field_encoding(session, _encoding_config())

    assert model.cell_ids.tolist() == [first, second]
    _assert_both_cells_received_spikes(model)


def test_position_mask_encoding_uses_exact_active_spike_selector() -> None:
    session, first, second = _large_id_session()

    assert (
        position_validation._spikes_and_cell_ids_for_encoding
        is encoding_module._spikes_and_cell_ids_for_encoding
    )
    model = position_validation.fit_place_field_encoding_for_position_mask(
        session,
        np.ones(session.position.shape[0], dtype=bool),
        _encoding_config(),
    )

    assert model.cell_ids.tolist() == [first, second]
    _assert_both_cells_received_spikes(model)
