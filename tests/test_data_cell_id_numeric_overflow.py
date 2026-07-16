from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.data import _mark_group_ids_from_tetrode_cell_ids


def test_mark_group_ids_normalizes_unrepresentable_numeric_ids() -> None:
    with pytest.raises(
        ValueError,
        match="tetrode/cell IDs must contain numeric integer identifiers",
    ):
        _mark_group_ids_from_tetrode_cell_ids(
            np.array([1], dtype=int),
            np.array([[10**400, 1]], dtype=object),
        )
