from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.position_validation import _spike_counts_for_window


def test_spike_counts_for_window_maps_unsorted_encoding_cell_ids():
    session = SimpleNamespace(
        spikes=np.array([[1.10, 20.0], [1.20, 10.0], [1.30, 20.0], [1.40, 99.0]], dtype=float),
        excitatory_neurons=np.array([], dtype=int),
    )
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((2, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([20, 10], dtype=int),
        config=EncodingConfig(),
    )

    np.testing.assert_array_equal(
        _spike_counts_for_window(session, encoding, 1.0, 2.0),
        np.array([2, 1], dtype=int),
    )
