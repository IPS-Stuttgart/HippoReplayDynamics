import numpy as np
import pytest

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
