from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import ground_truth as gt


def test_score_table_cell_ids_accept_integral_float_text() -> None:
    np.testing.assert_array_equal(
        gt._parse_cell_ids("[1.0, 2, 3]"),
        np.array([1, 2, 3], dtype=int),
    )


def test_score_table_cell_ids_treat_missing_text_as_absent() -> None:
    for value in ("", "nan", "NaN", "None", "null", "<NA>", "[]"):
        assert gt._parse_cell_ids(value) is None


def test_score_table_cell_ids_reject_fractional_text() -> None:
    with pytest.raises(ValueError, match="score-table cell IDs"):
        gt._parse_cell_ids("1 2.5 3")


def test_score_table_cell_ids_reject_fractional_array() -> None:
    with pytest.raises(ValueError, match="score-table cell IDs"):
        gt._parse_cell_ids(np.array([1.0, 2.25, 3.0], dtype=float))


def test_score_table_cell_ids_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="score-table cell IDs"):
        gt._parse_cell_ids("1 inf 3")
