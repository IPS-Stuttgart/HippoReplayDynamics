from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.shuffle_controls import shuffled_encoding


def _two_cell_two_bin_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([10, 11], dtype=int),
        config=EncodingConfig(),
    )


def test_cell_permutation_rejects_identity_draw() -> None:
    encoding = _two_cell_two_bin_encoding()

    control = shuffled_encoding(
        encoding,
        mode="cell-permutation",
        random_seed=0,
    )

    np.testing.assert_array_equal(
        control.rates_hz,
        encoding.rates_hz[[1, 0]],
    )


def test_spatial_permutation_rejects_identity_draw() -> None:
    encoding = _two_cell_two_bin_encoding()

    control = shuffled_encoding(
        encoding,
        mode="spatial-permutation",
        random_seed=0,
    )

    np.testing.assert_array_equal(
        control.rates_hz,
        encoding.rates_hz[:, [1, 0]],
    )


def test_independent_spatial_permutation_rejects_identity_per_cell() -> None:
    encoding = _two_cell_two_bin_encoding()

    control = shuffled_encoding(
        encoding,
        mode="independent-spatial-permutation",
        random_seed=0,
    )

    np.testing.assert_array_equal(
        control.rates_hz,
        encoding.rates_hz[:, [1, 0]],
    )


def test_permutation_modes_preserve_unavoidable_singletons() -> None:
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.array([[2.5]], dtype=float),
        occupancy_s=np.ones(1, dtype=float),
        cell_ids=np.array([10], dtype=int),
        config=EncodingConfig(),
    )

    for mode in (
        "cell-permutation",
        "spatial-permutation",
        "independent-spatial-permutation",
    ):
        control = shuffled_encoding(
            encoding,
            mode=mode,
            random_seed=0,
        )
        np.testing.assert_array_equal(control.rates_hz, encoding.rates_hz)
