from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.result_improvements import stratified_cell_split


def _scores(count: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, count)


@pytest.mark.parametrize(
    "cell_ids",
    [
        np.array([1.5, 2.0, 3.0]),
        np.array([True, False, True]),
        np.array([1, 1, 2]),
    ],
    ids=["fractional", "boolean", "duplicate"],
)
def test_stratified_cell_split_rejects_malformed_cell_ids(
    cell_ids: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="cell_ids"):
        stratified_cell_split(
            cell_ids,
            _scores(cell_ids.size),
            1.0 / 3.0,
            1,
        )


def test_stratified_cell_split_preserves_integral_float_cell_ids() -> None:
    train, test = stratified_cell_split(
        np.array([10.0, 20.0, 30.0, 40.0]),
        _scores(4),
        0.25,
        1,
        n_strata=2,
    )

    np.testing.assert_array_equal(
        np.sort(np.concatenate([train, test])),
        np.array([10, 20, 30, 40]),
    )
    assert np.intersect1d(train, test).size == 0
