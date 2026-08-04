from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from hipporeplayimm import benchmark_cell_split_metadata
from hipporeplayimm.ground_truth_integer_metadata import (
    apply_ground_truth_integer_metadata_patch,
)


def _strata_frame(value: object) -> pd.DataFrame:
    return pd.DataFrame({"benchmark_cell_split_strata": pd.Series([value], dtype=object)})


@pytest.mark.parametrize(
    "value",
    [
        "4.0000000001",
        "3.9999999999",
        Decimal("4.0000000001"),
        Decimal("3.9999999999"),
    ],
)
def test_cell_split_strata_reject_fractional_score_metadata(value: object) -> None:
    apply_ground_truth_integer_metadata_patch()

    with pytest.raises(
        ValueError,
        match="benchmark_cell_split_strata must contain integer values",
    ):
        benchmark_cell_split_metadata._cell_split_strata_from_scores(
            _strata_frame(value),
            4,
        )


@pytest.mark.parametrize("value", [4, 4.0, "4.0", "4e0", Decimal("4")])
def test_cell_split_strata_accept_exact_integer_score_metadata(value: object) -> None:
    apply_ground_truth_integer_metadata_patch()

    assert (
        benchmark_cell_split_metadata._cell_split_strata_from_scores(
            _strata_frame(value),
            2,
        )
        == 4
    )


def test_cell_split_strata_preserve_large_exact_integer_text() -> None:
    apply_ground_truth_integer_metadata_patch()
    large_value = 2**53 + 1

    assert (
        benchmark_cell_split_metadata._cell_split_strata_from_scores(
            _strata_frame(str(large_value)),
            4,
        )
        == large_value
    )
