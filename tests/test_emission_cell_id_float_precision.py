from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.emission_cell_id_validation import _coerce_integral_ids


@pytest.mark.parametrize("value", [float(2**53), -float(2**53)])
def test_cell_id_validation_rejects_unsafe_float_identifiers(value: float) -> None:
    with pytest.raises(ValueError, match=r"at or above 2\*\*53 are unsafe"):
        _coerce_integral_ids(np.asarray([value], dtype=float), "cell IDs")


def test_cell_id_validation_keeps_exact_large_nonfloat_identifiers() -> None:
    value = 2**53 + 1

    integer_ids = _coerce_integral_ids(np.asarray([value], dtype=object), "cell IDs")
    string_ids = _coerce_integral_ids(np.asarray([str(value)], dtype=object), "cell IDs")

    assert integer_ids.tolist() == [value]
    assert string_ids.tolist() == [value]


def test_cell_id_validation_still_accepts_safe_integral_float() -> None:
    value = float(2**53 - 1)

    ids = _coerce_integral_ids(np.asarray([value], dtype=float), "cell IDs")

    assert ids.tolist() == [2**53 - 1]
