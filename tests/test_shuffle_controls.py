from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.shuffle_controls import shuffled_encoding


def test_independent_spatial_permutation_handles_empty_cell_set():
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.empty((0, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([], dtype=int),
        config=EncodingConfig(),
    )

    control = shuffled_encoding(
        encoding,
        mode="independent-spatial-permutation",
        random_seed=7,
    )

    assert control.rates_hz.shape == (0, 2)
    assert control.rates_hz.dtype == float
    np.testing.assert_array_equal(control.cell_ids, np.array([], dtype=int))
    np.testing.assert_allclose(control.occupancy_s, np.ones(2, dtype=float))
