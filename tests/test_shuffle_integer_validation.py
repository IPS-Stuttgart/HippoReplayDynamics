from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.shuffle_controls import _nonnegative_integer_value


@pytest.mark.parametrize(
    "value",
    ["4", b"4", np.str_("4"), np.bytes_(b"4"), np.asarray("4"), np.asarray(b"4")],
)
def test_shuffle_integer_validation_rejects_string_values(value: object) -> None:
    with pytest.raises(ValueError, match="random_seed.*string"):
        _nonnegative_integer_value("random_seed", value)


def test_shuffle_integer_validation_preserves_large_uint64_values() -> None:
    value = np.uint64(2**63 + 123)

    assert _nonnegative_integer_value("random_seed", value) == int(value)


def test_shuffle_integer_validation_preserves_decimal_seed_exactly() -> None:
    seed = Decimal(2**53 + 1)

    assert _nonnegative_integer_value("random_seed", seed) == int(seed)


def test_shuffle_integer_validation_rejects_fractional_decimal_values() -> None:
    value = Decimal("9007199254740992.5")

    with pytest.raises(ValueError, match="random_seed must be an integer"):
        _nonnegative_integer_value("random_seed", value)


def test_shuffle_integer_validation_preserves_extended_precision_seed() -> None:
    seed = np.longdouble(str(2**53 + 1))
    if int(seed) != 2**53 + 1:
        pytest.skip("numpy.longdouble does not exceed binary64 integer precision")

    assert _nonnegative_integer_value("random_seed", seed) == 2**53 + 1


def test_shuffle_integer_validation_rejects_fractional_extended_precision() -> None:
    value = np.longdouble(2**53) + np.longdouble("0.5")
    if value == np.floor(value):
        pytest.skip("numpy.longdouble does not retain the fractional test value")

    with pytest.raises(ValueError, match="random_seed must be an integer"):
        _nonnegative_integer_value("random_seed", value)
