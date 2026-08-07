from __future__ import annotations

import numpy as np

import hipporeplayimm
from hipporeplayimm.accuracy_upgrades import restrict_encoding_to_mask
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def _two_by_two_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        bin_centers=np.array(
            [
                [0.5, 0.5],
                [0.5, 1.5],
                [1.5, 0.5],
                [1.5, 1.5],
            ],
            dtype=float,
        ),
        rates_hz=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float),
        occupancy_s=np.ones(4, dtype=float),
        cell_ids=np.array([7], dtype=int),
        config=EncodingConfig(bin_size_cm=1.0),
    )


def test_restricted_encoding_maps_positions_to_compact_bin_indices() -> None:
    hipporeplayimm.apply_runtime_patches()
    encoding = _two_by_two_encoding()

    restricted = restrict_encoding_to_mask(
        encoding,
        np.array([False, True, True, False], dtype=bool),
    )

    assert restricted.n_bins == 2
    np.testing.assert_array_equal(
        restricted.positions_to_flat_bins(encoding.bin_centers),
        np.array([-1, 0, 1, -1], dtype=int),
    )


def test_nested_restriction_and_cell_selection_preserve_compact_mapping() -> None:
    hipporeplayimm.apply_runtime_patches()
    encoding = _two_by_two_encoding()
    restricted = restrict_encoding_to_mask(
        encoding,
        np.array([False, True, True, False], dtype=bool),
    )

    nested = restrict_encoding_to_mask(
        restricted,
        np.array([False, True], dtype=bool),
    )
    selected = nested.select_cells([7])

    assert nested.n_bins == 1
    np.testing.assert_array_equal(
        nested.positions_to_flat_bins(encoding.bin_centers),
        np.array([-1, -1, 0, -1], dtype=int),
    )
    np.testing.assert_array_equal(
        selected.positions_to_flat_bins(encoding.bin_centers),
        np.array([-1, -1, 0, -1], dtype=int),
    )
