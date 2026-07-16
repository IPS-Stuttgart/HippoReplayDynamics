from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.encoding as encoding


def _extended_precision_integer() -> tuple[np.longdouble, int]:
    expected = 2**53 + 1
    value = np.longdouble(str(expected))
    if int(value) != expected:
        pytest.skip("platform longdouble does not exceed binary64 precision")
    return value, expected


def test_active_emission_row_lookup_preserves_large_object_integer_ids() -> None:
    first = 2**53
    second = first + 1

    rows = encoding._cell_id_row_indices(
        np.array([first, second], dtype=object),
        np.array([second], dtype=object),
    )

    assert rows.tolist() == [1]


def test_active_emission_row_lookup_preserves_extended_precision_ids() -> None:
    second, expected = _extended_precision_integer()
    first = np.longdouble(str(expected - 1))

    rows = encoding._cell_id_row_indices(
        np.array([first, second], dtype=np.longdouble),
        np.array([second], dtype=np.longdouble),
    )

    assert rows.tolist() == [1]


def test_active_emission_row_lookup_rejects_fractional_extended_precision_ids() -> None:
    fractional = np.longdouble("9007199254740992.5")
    if fractional.is_integer():
        pytest.skip("platform longdouble does not exceed binary64 precision")

    with pytest.raises(ValueError, match="integer-valued identifiers"):
        encoding._cell_id_row_indices(
            np.array([fractional], dtype=np.longdouble),
            np.array([fractional], dtype=np.longdouble),
        )
