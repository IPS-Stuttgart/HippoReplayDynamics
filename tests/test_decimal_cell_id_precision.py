from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.emission_cell_id_validation import _cell_id_row_indices


def test_large_decimal_emission_cell_ids_remain_distinct() -> None:
    first = Decimal(2**53)
    second = first + 1

    rows = _cell_id_row_indices(
        np.array([first, second], dtype=object),
        np.array([second], dtype=object),
    )

    assert rows.tolist() == [1]


@pytest.mark.parametrize("value", [Decimal("1.5"), Decimal("NaN"), Decimal("Infinity")])
def test_invalid_decimal_emission_cell_ids_are_rejected(value: Decimal) -> None:
    with pytest.raises(ValueError, match="finite integer|integer-valued"):
        _cell_id_row_indices(
            np.array([Decimal(1)], dtype=object),
            np.array([value], dtype=object),
        )
