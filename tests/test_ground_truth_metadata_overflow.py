from __future__ import annotations

import pytest

from hipporeplayimm.ground_truth_cell_id_metadata import _parse_cell_id_value
from hipporeplayimm.ground_truth_integer_metadata import _parse_integer_metadata_value


class _OverflowingNumeric:
    def __float__(self) -> float:
        raise OverflowError("too large to convert to float")


def test_ground_truth_integer_metadata_overflow_is_value_error() -> None:
    with pytest.raises(ValueError, match="event_index.*integer values"):
        _parse_integer_metadata_value("event_index", _OverflowingNumeric())


def test_ground_truth_cell_id_metadata_overflow_is_value_error() -> None:
    with pytest.raises(ValueError, match="cell IDs.*integer values"):
        _parse_cell_id_value(_OverflowingNumeric())
