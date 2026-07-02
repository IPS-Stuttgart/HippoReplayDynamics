from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.emission_cell_id_validation import _cell_id_row_indices


def test_large_integer_log_emission_cell_ids_remain_distinct() -> None:
    first = 2**53
    second = first + 1

    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 2), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.02,
        cell_ids=np.array([first, second], dtype=object),
        n_spikes=0,
    )

    assert emissions.cell_ids.tolist() == [first, second]


def test_large_integer_emission_row_lookup_remains_distinct() -> None:
    first = 2**53
    second = first + 1

    rows = _cell_id_row_indices(
        np.array([first, second], dtype=object),
        np.array([second], dtype=object),
    )

    assert rows.tolist() == [1]
