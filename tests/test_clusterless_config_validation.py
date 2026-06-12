from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, fit_clusterless_mark_encoding
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EncodingConfig


def _marked_session() -> ReplaySession:
    spike_times = np.array([0.5], dtype=float)
    return ReplaySession(
        rat="RatX",
        name="OpenY",
        path=Path("unused"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 10.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.array([[0.5, 1.0]], dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([1], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 1.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=spike_times,
            marks=np.array([[10.0, 20.0]], dtype=float),
            source_file="Spike_Data.mat",
            source_variable="Spike_Marks",
            feature_names=("mark_0", "mark_1"),
            cell_ids=np.array([1], dtype=int),
        ),
    )


def test_clusterless_mark_encoding_validates_nested_encoding_config() -> None:
    session = _marked_session()
    config = ClusterlessMarkConfig(
        encoding=EncodingConfig(min_speed_cm_s=0.0, min_occupancy_s=0.0),
    )

    with pytest.raises(ValueError, match="min_occupancy_s"):
        fit_clusterless_mark_encoding(session, config)
