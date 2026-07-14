from __future__ import annotations

import argparse

import pytest

from scripts.accuracy_upgrade_report import _nonnegative_int


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("3", 3), ("+4", 4)])
def test_nonnegative_int_accepts_nonnegative_values(value: str, expected: int) -> None:
    assert _nonnegative_int(value) == expected


@pytest.mark.parametrize("value", ["-1", "1.5", "not-an-integer"])
def test_nonnegative_int_rejects_negative_or_malformed_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative integer"):
        _nonnegative_int(value)
