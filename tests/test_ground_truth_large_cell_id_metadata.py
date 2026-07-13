from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.ground_truth import _parse_cell_ids


@pytest.mark.parametrize(
    "value",
    [
        "[9007199254740992 9007199254740993]",
        "[9007199254740992.0 9007199254740993.0]",
        np.array([9007199254740992, 9007199254740993], dtype=object),
        np.array([b"9007199254740992", b"9007199254740993"], dtype=object),
    ],
)
def test_ground_truth_cell_id_metadata_preserves_large_integer_identity(
    value: object,
) -> None:
    parsed = _parse_cell_ids(value)

    assert parsed is not None
    assert parsed.tolist() == [2**53, 2**53 + 1]
