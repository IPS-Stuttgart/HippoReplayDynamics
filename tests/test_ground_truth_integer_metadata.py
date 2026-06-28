import pandas as pd
import pytest

from hipporeplayimm.ground_truth import _parse_cell_ids, _unique_int_from_column


def test_unique_int_from_column_accepts_integer_valued_metadata():
    frame = pd.DataFrame({"benchmark_random_seed": ["7", "7.0"]})

    assert _unique_int_from_column(frame, "benchmark_random_seed", 1) == 7


def test_unique_int_from_column_rejects_fractional_metadata():
    frame = pd.DataFrame({"benchmark_random_seed": ["7.5"]})

    with pytest.raises(ValueError, match="benchmark_random_seed"):
        _unique_int_from_column(frame, "benchmark_random_seed", 1)


def test_unique_int_from_column_rejects_near_integral_metadata():
    frame = pd.DataFrame({"benchmark_random_seed": ["7.0000000005"]})

    with pytest.raises(ValueError, match="benchmark_random_seed"):
        _unique_int_from_column(frame, "benchmark_random_seed", 1)


def test_unique_int_from_column_rejects_boolean_metadata():
    frame = pd.DataFrame({"benchmark_random_seed": [True]})

    with pytest.raises(ValueError, match="benchmark_random_seed"):
        _unique_int_from_column(frame, "benchmark_random_seed", 1)


def test_parse_cell_ids_rejects_near_integral_metadata():
    with pytest.raises(ValueError, match="score-table cell IDs"):
        _parse_cell_ids("[1.0000000005 2]")
