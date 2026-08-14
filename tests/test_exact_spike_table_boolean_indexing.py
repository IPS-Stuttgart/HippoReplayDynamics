from __future__ import annotations

import numpy as np

from hipporeplayimm.spike_cell_id_emission_validation import _exact_spike_table


def test_exact_spike_table_preserves_boolean_scalar_indexing() -> None:
    spikes = np.array([[0.25, 7], [0.75, 8]], dtype=object)
    table = _exact_spike_table(spikes, np.array([7, 8], dtype=int))

    selected = table[:, True]

    assert selected.shape == (2, 1, 2)
    assert selected.dtype == object
    assert selected[0, 0, 0] == 0.25
    assert selected[1, 0, 0] == 0.75
    assert selected[0, 0, 1] == 7
    assert selected[1, 0, 1] == 8


def test_exact_spike_table_preserves_numpy_boolean_scalar_indexing() -> None:
    spikes = np.array([[0.25, 7], [0.75, 8]], dtype=object)
    table = _exact_spike_table(spikes, np.array([7, 8], dtype=int))

    selected = table[:, np.bool_(False)]

    assert selected.shape == (2, 0, 2)
    assert selected.dtype == object
