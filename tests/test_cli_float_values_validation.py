from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm import cli
from hipporeplayimm.cli_float_values_validation import _positive_integer_count


class _OverflowingNumeric:
    def __float__(self) -> float:
        raise OverflowError("too large to convert to float")


def test_cli_float_value_grids_reject_nonfinite_values() -> None:
    for raw in ("nan", "1.0,nan", "inf", "-inf"):
        with pytest.raises(ValueError, match="finite"):
            cli._parse_float_values(raw)


def test_cli_float_value_grids_accept_finite_values() -> None:
    assert cli._parse_float_values("0.5, 1, 2.5") == (0.5, 1.0, 2.5)


def test_positive_integer_count_overflow_is_value_error() -> None:
    with pytest.raises(ValueError, match="n_bootstrap.*positive integer"):
        _positive_integer_count("n_bootstrap", _OverflowingNumeric())


@pytest.mark.parametrize("value", [2**53 + 1, np.uint64(2**53 + 1)])
def test_positive_integer_count_preserves_large_integers(value) -> None:
    assert _positive_integer_count("n_bootstrap", value) == 2**53 + 1


def test_positive_integer_count_preserves_large_decimal_integer() -> None:
    value = Decimal(2**53 + 1)

    assert _positive_integer_count("n_bootstrap", value) == 2**53 + 1


@pytest.mark.parametrize("name", ["n_bootstrap", "n_permutations"])
def test_positive_integer_count_rejects_lossy_fractional_decimal(name: str) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        _positive_integer_count(name, Decimal("9007199254740992.5"))
