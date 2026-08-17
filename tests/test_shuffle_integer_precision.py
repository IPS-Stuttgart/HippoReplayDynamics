from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.shuffle_controls import _nonnegative_integer_value


def test_nonnegative_integer_value_preserves_decimal_beyond_binary64_precision() -> None:
    expected = 2**53 + 1

    resolved = _nonnegative_integer_value("random_seed", Decimal(expected))

    assert resolved == expected


def test_nonnegative_integer_value_preserves_extended_precision_numpy_scalar() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("platform longdouble does not exceed float64 precision")
    expected = 2**53 + 1
    value = np.longdouble(2**53) + np.longdouble(1)

    resolved = _nonnegative_integer_value("random_seed", value)

    assert resolved == expected
