from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.data_cell_id_validation import _coerce_ripple_index


def _nested_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def test_decimal_ripple_index_terminates_and_preserves_integrality() -> None:
    assert _coerce_ripple_index(Decimal("1"), 2) == 1
    assert _coerce_ripple_index(_nested_scalar(Decimal("1")), 2) == 1


def test_fractional_decimal_ripple_index_does_not_alias_through_float() -> None:
    value = Decimal("1.0000000000000000000000000001")
    assert float(value) == 1.0

    with pytest.raises(TypeError, match="ripple index must be an integer"):
        _coerce_ripple_index(value, 2)

    with pytest.raises(TypeError, match="ripple index must be an integer"):
        _coerce_ripple_index(_nested_scalar(value), 2)


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_nonfinite_decimal_ripple_indices_are_rejected(value: Decimal) -> None:
    with pytest.raises(TypeError, match="ripple index must be an integer"):
        _coerce_ripple_index(value, 2)
