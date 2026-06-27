from __future__ import annotations

import pandas as pd
import pytest

import hipporeplayimm.score_metadata as score_metadata_module
from hipporeplayimm.benchmarks import _coerce_bool_series as _benchmark_coerce_bool_series
from hipporeplayimm.evidence_reporting import _coerce_bool_series
from hipporeplayimm.ground_truth_float_metadata import _parse_bool_metadata_value
from hipporeplayimm.score_metadata import _parse_bool as _parse_score_bool
from hipporeplayimm.score_metadata_bool_validation import apply_score_metadata_bool_validation_patch


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


def test_score_metadata_bool_patch_refreshes_stale_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    def stale_parse_bool(_value: object) -> bool:
        return False

    monkeypatch.setattr(score_metadata_module, "_parse_bool", stale_parse_bool)
    monkeypatch.setattr(
        score_metadata_module,
        "_score_metadata_bool_validation_patch_applied",
        True,
        raising=False,
    )

    apply_score_metadata_bool_validation_patch()

    assert score_metadata_module._parse_bool("1") is True
    with pytest.raises(ValueError, match="boolean value"):
        score_metadata_module._parse_bool("2")


def test_evidence_bool_series_does_not_admit_nonbinary_numeric_values() -> None:
    parsed = _coerce_bool_series(pd.Series(["2", "2.0", 2, -1, 0.5]), default=False)

    assert parsed.tolist() == [False, False, False, False, False]


def test_evidence_bool_series_keeps_binary_numeric_values() -> None:
    parsed = _coerce_bool_series(pd.Series(["1", "1.0", 1, "0", "0.0", 0]), default=False)

    assert parsed.tolist() == [True, True, True, False, False, False]


def test_preimported_benchmark_bool_series_alias_is_synchronized() -> None:
    parsed = _benchmark_coerce_bool_series(pd.Series(["2", "1"]), default=False)

    assert parsed.tolist() == [False, True]
