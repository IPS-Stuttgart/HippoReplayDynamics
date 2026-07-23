from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.ground_truth_cell_id_metadata import _parse_cell_id_value
from hipporeplayimm.ground_truth_integer_metadata import _parse_integer_metadata_value


class _OverflowingNumeric:
    def __float__(self) -> float:
        raise OverflowError("too large to convert to float")


def test_ground_truth_integer_metadata_overflow_is_value_error() -> None:
    with pytest.raises(ValueError, match="event_index.*integer values"):
        _parse_integer_metadata_value("event_index", _OverflowingNumeric())


def test_ground_truth_integer_metadata_preserves_large_integer_strings() -> None:
    large_value = 2**53 + 1

    assert _parse_integer_metadata_value("event_index", str(large_value)) == large_value


def test_ground_truth_integer_metadata_preserves_extended_precision_values() -> None:
    value = np.longdouble(str(2**53 + 1))
    if int(value) != 2**53 + 1:
        pytest.skip("numpy.longdouble does not exceed binary64 integer precision")

    assert _parse_integer_metadata_value("event_index", value) == 2**53 + 1


def test_ground_truth_integer_metadata_rejects_fractional_extended_precision() -> None:
    value = np.longdouble(2**53) + np.longdouble("0.5")
    if value == np.floor(value):
        pytest.skip("numpy.longdouble does not retain the fractional test value")

    with pytest.raises(ValueError, match="event_index.*integer values"):
        _parse_integer_metadata_value("event_index", value)


def test_ground_truth_cell_id_metadata_overflow_is_value_error() -> None:
    with pytest.raises(ValueError, match="cell IDs.*integer values"):
        _parse_cell_id_value(_OverflowingNumeric())
