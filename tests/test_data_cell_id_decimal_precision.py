from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.data_cell_id_validation import _coerce_integral_ids


def test_large_integral_decimal_cell_id_is_preserved_exactly() -> None:
    value = Decimal("9007199254740993")

    result = _coerce_integral_ids(
        np.array([value], dtype=object),
        "spike cell IDs",
    )

    np.testing.assert_array_equal(
        result,
        np.array([9007199254740993], dtype=int),
    )


def test_large_fractional_decimal_cell_id_is_rejected_without_float_rounding() -> None:
    value = Decimal("9007199254740992.5")

    with pytest.raises(ValueError, match="integer-valued"):
        _coerce_integral_ids(
            np.array([value], dtype=object),
            "spike cell IDs",
        )
