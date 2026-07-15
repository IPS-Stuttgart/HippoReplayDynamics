from decimal import Decimal

import pytest

from hipporeplayimm.sweep_seed_validation import _seed_value


def test_seed_value_parses_large_text_seed_exactly():
    seed = 9_007_199_254_740_993

    assert _seed_value(str(seed), "random_seed") == seed
    assert _seed_value(str(seed).encode("ascii"), "random_seed") == seed


def test_seed_value_rejects_fractional_large_text_seed():
    with pytest.raises(ValueError, match="finite nonnegative integer"):
        _seed_value("9007199254740993.5", "random_seed")


def test_seed_value_preserves_large_decimal_seed_exactly():
    seed = 9_007_199_254_740_993

    assert _seed_value(Decimal(seed), "random_seed") == seed


def test_seed_value_rejects_fractional_decimal_seed():
    with pytest.raises(ValueError, match="finite nonnegative integer"):
        _seed_value(Decimal("9007199254740993.5"), "random_seed")
