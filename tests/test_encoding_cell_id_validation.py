from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, fit_place_field_encoding


def _session(tmp_path, cell_id):
    times = np.array([0.0, 1.0, 2.0], dtype=float)
    position = np.column_stack([times, [0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=position,
        spikes=np.array([[0.5, cell_id]], dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6)),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


def test_encoding_rejects_fractional_cell_ids(tmp_path):
    with pytest.raises(ValueError, match="cell IDs"):
        fit_place_field_encoding(_session(tmp_path, 1.5), EncodingConfig(min_speed_cm_s=0.0))


def test_encoding_accepts_integral_float_cell_ids(tmp_path):
    encoding = fit_place_field_encoding(_session(tmp_path, 1.0), EncodingConfig(min_speed_cm_s=0.0))
    assert encoding.cell_ids.tolist() == [1]
