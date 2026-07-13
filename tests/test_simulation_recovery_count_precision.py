from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.simulation_recovery_count_validation import _positive_integer_scalar


@pytest.mark.parametrize(
    "value",
    [
        2**53 + 1,
        np.int64(2**53 + 1),
        Decimal("9007199254740993"),
    ],
)
def test_positive_integer_scalar_preserves_exact_values_above_float_precision(
    value: object,
) -> None:
    assert _positive_integer_scalar("n_time", value) == 2**53 + 1


@pytest.mark.parametrize("value", [0, -1, 1.5, Decimal("1.5")])
def test_positive_integer_scalar_still_rejects_nonpositive_or_fractional_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="n_time must be a positive integer"):
        _positive_integer_scalar("n_time", value)
