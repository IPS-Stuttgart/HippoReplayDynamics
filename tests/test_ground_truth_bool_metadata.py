from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.evidence_reporting import _coerce_bool_series
from hipporeplayimm.ground_truth_float_metadata import _parse_bool_metadata_value
from hipporeplayimm.score_metadata import _parse_bool as _parse_score_bool


def test_parse_bool_metadata_rejects_nonbinary_numeric_strings() -> None:
    for raw in ("2", "2.0", "-1", "0.5"):
        with pytest.raises(ValueError, match="boolean values"):
            _parse_bool_metadata_value("metadata_flag", raw)


def test_parse_bool_metadata_keeps_binary_numeric_strings() -> None:
    assert _parse_bool_metadata_value("metadata_flag", "1") is True
    assert _parse_bool_metadata_value("metadata_flag", "1.0") is True
    assert _parse_bool_metadata_value("metadata_flag", "0") is False
    assert _parse_bool_metadata_value("metadata_flag", "0.0") is False


def test_score_metadata_bool_rejects_nonbinary_numeric_values() -> None:
    for raw in ("2", "2.0", -1, 0.5):
        with pytest.raises(ValueError, match="boolean values"):
            _parse_score_bool(raw)


def test_score_metadata_bool_keeps_binary_numeric_values() -> None:
    assert _parse_score_bool("1") is True
    assert _parse_score_bool(1) is True
    assert _parse_score_bool("0") is False
    assert _parse_score_bool(0) is False


def test_evidence_bool_series_does_not_admit_nonbinary_numeric_values() -> None:
    parsed = _coerce_bool_series(pd.Series(["2", "2.0", 2, -1, 0.5]), default=False)

    assert parsed.tolist() == [False, False, False, False, False]


def test_evidence_bool_series_keeps_binary_numeric_values() -> None:
    parsed = _coerce_bool_series(pd.Series(["1", "1.0", 1, "0", "0.0", 0]), default=False)

    assert parsed.tolist() == [True, True, True, False, False, False]
