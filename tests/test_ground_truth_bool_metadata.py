from __future__ import annotations

import pytest

from hipporeplayimm.ground_truth_float_metadata import _parse_bool_metadata_value


def test_parse_bool_metadata_rejects_nonbinary_numeric_strings() -> None:
    for raw in ("2", "2.0", "-1", "0.5"):
        with pytest.raises(ValueError, match="boolean values"):
            _parse_bool_metadata_value("metadata_flag", raw)


def test_parse_bool_metadata_keeps_binary_numeric_strings() -> None:
    assert _parse_bool_metadata_value("metadata_flag", "1") is True
    assert _parse_bool_metadata_value("metadata_flag", "1.0") is True
    assert _parse_bool_metadata_value("metadata_flag", "0") is False
    assert _parse_bool_metadata_value("metadata_flag", "0.0") is False
