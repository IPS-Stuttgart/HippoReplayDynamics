from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.data_cell_id_validation import _coerce_integral_ids, _coerce_ripple_index


def _fractional_extended_float() -> np.longdouble:
    value = np.nextafter(np.longdouble(1.0), np.longdouble(2.0))
    if bool(np.equal(value, np.trunc(value))) or not float(value).is_integer():
        pytest.skip("platform longdouble does not expose the binary64 narrowing case")
    return value


def test_cell_ids_reject_fractional_extended_precision_values():
    value = _fractional_extended_float()

    with pytest.raises(ValueError, match="integer-valued"):
        _coerce_integral_ids(
            np.asarray([value], dtype=np.longdouble),
            "spike cell IDs",
        )


def test_ripple_indices_reject_fractional_extended_precision_values():
    value = _fractional_extended_float()

    with pytest.raises(TypeError, match="ripple index must be an integer"):
        _coerce_ripple_index(value, 2)


@pytest.mark.parametrize("imaginary", [0.0, 2.0])
def test_cell_ids_reject_extended_precision_complex_values(imaginary):
    value = np.clongdouble(1.0 + imaginary * 1j)

    with pytest.raises(ValueError, match="real integer identifiers"):
        _coerce_integral_ids(
            np.asarray([value], dtype=np.clongdouble),
            "spike cell IDs",
        )


def test_ripple_indices_reject_extended_precision_complex_values():
    value = np.clongdouble(1.0 + 2.0j)

    with pytest.raises(TypeError, match="ripple index must be an integer"):
        _coerce_ripple_index(value, 2)


def test_integral_extended_precision_values_remain_supported():
    values = _coerce_integral_ids(
        np.asarray([np.longdouble(1.0), np.longdouble(2.0)]),
        "spike cell IDs",
    )

    np.testing.assert_array_equal(values, np.asarray([1, 2], dtype=int))
    assert _coerce_ripple_index(np.longdouble(1.0), 2) == 1
