from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.result_improvements import stratified_cell_split


def _scores(count: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, count)


def test_stratified_cell_split_rejects_invalid_n_strata() -> None:
    cells = np.arange(8)
    scores = np.linspace(0.0, 1.0, cells.size)

    for invalid in (True, np.array(True), 0, -1, 1.5, np.nan, np.array([4])):
        with pytest.raises(ValueError, match="n_strata must be a positive integer"):
            stratified_cell_split(cells, scores, 0.25, 1, n_strata=invalid)


def test_stratified_cell_split_accepts_integral_float_n_strata() -> None:
    cells = np.arange(8)
    scores = np.linspace(0.0, 1.0, cells.size)

    train, test = stratified_cell_split(cells, scores, 0.25, 1, n_strata=4.0)

    assert train.size + test.size == cells.size
    assert test.size == 2
    assert np.intersect1d(train, test).size == 0


@pytest.mark.parametrize(
    "random_seed",
    [
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(False), id="numpy-bool"),
        pytest.param(np.array([1]), id="one-dimensional-array"),
        pytest.param("1", id="numeric-string"),
        pytest.param(1.5, id="fractional"),
        pytest.param(-1, id="negative"),
        pytest.param(np.nan, id="nan"),
    ],
)
def test_stratified_cell_split_rejects_malformed_random_seed(
    random_seed: object,
) -> None:
    cells = np.arange(8)

    with pytest.raises(
        ValueError,
        match="random_seed must be a finite nonnegative integer",
    ):
        stratified_cell_split(
            cells,
            _scores(cells.size),
            0.25,
            random_seed,
        )


def test_stratified_cell_split_normalizes_integral_scalar_seed() -> None:
    cells = np.arange(12)
    scores = _scores(cells.size)

    expected = stratified_cell_split(cells, scores, 0.25, 7)
    actual = stratified_cell_split(cells, scores, 0.25, np.array(7.0))

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize(
    "cell_ids",
    [
        pytest.param(np.array([1.5, 2.0, 3.0]), id="fractional"),
        pytest.param(np.array([True, False, True]), id="boolean"),
        pytest.param(np.array([1, 1, 2]), id="duplicate"),
    ],
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
