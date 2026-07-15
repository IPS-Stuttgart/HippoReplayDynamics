import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import data as data_module
from hipporeplayimm.data import _mark_group_ids_from_tetrode_cell_ids


def test_tetrode_mapping_rejects_conflicting_duplicate_cell_rows() -> None:
    with pytest.raises(ValueError, match=r"cell ID 11 maps to multiple tetrode/group IDs"):
        _mark_group_ids_from_tetrode_cell_ids(
            np.array([11, 12, 11], dtype=int),
            np.array(
                [
                    [7, 11],
                    [8, 11],
                    [9, 12],
                ],
                dtype=int,
            ),
        )


def test_tetrode_mapping_allows_consistent_duplicate_cell_rows() -> None:
    group_ids = _mark_group_ids_from_tetrode_cell_ids(
        np.array([11, 12, 11], dtype=int),
        np.array(
            [
                [7, 11],
                [7, 11],
                [9, 12],
            ],
            dtype=int,
        ),
    )

    assert group_ids is not None
    np.testing.assert_array_equal(group_ids, np.array([7, 9, 7], dtype=int))


def test_tetrode_mapping_resolves_equal_overlap_by_functional_orientation() -> None:
    group_ids = _mark_group_ids_from_tetrode_cell_ids(
        np.array([1, 2, 1, 2], dtype=int),
        np.array(
            [
                [1, 1],
                [2, 1],
                [3, 2],
                [4, 2],
            ],
            dtype=int,
        ),
    )

    assert group_ids is not None
    np.testing.assert_array_equal(group_ids, np.array([1, 1, 1, 1], dtype=int))


def test_tetrode_mapping_rejects_unresolved_equal_overlap_orientation() -> None:
    with pytest.raises(ValueError, match="ambiguous Tetrode_Cell_IDs orientation"):
        _mark_group_ids_from_tetrode_cell_ids(
            np.array([1, 2, 3, 4], dtype=int),
            np.array(
                [
                    [1, 3],
                    [2, 4],
                ],
                dtype=int,
            ),
        )


def test_tetrode_mapping_orientation_patch_is_idempotent() -> None:
    current = data_module._mark_group_ids_from_tetrode_cell_ids

    hipporeplayimm.apply_runtime_patches()

    assert data_module._mark_group_ids_from_tetrode_cell_ids is current
