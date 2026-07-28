from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.simulation_recovery_count_validation import _positive_integer_scalar


@pytest.mark.parametrize(
    "value",
    [
        2**53 + 1,
        np.uint64(2**63 + 1),
        Decimal("9007199254740995"),
    ],
)
def test_positive_integer_scalar_preserves_large_integer_precision(value):
    assert _positive_integer_scalar("n_time", value) == int(value)


def test_positive_integer_scalar_rejects_fractional_decimal_above_float_precision():
    with pytest.raises(ValueError, match="n_time must be a positive integer"):
        _positive_integer_scalar("n_time", Decimal("9007199254740993.5"))
