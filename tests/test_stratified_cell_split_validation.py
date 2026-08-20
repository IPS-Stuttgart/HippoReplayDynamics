from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hipporeplayimm.result_improvement_split_validation import _synchronize_aliases
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


def test_split_alias_sync_ignores_similarly_named_top_level_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = object()
    patched = object()
    unrelated = types.ModuleType("hipporeplayimm_extension")
    package_child = types.ModuleType("hipporeplayimm._split_alias_probe")
    unrelated.stratified_cell_split = original
    package_child.stratified_cell_split = original
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)
    monkeypatch.setitem(sys.modules, package_child.__name__, package_child)

    _synchronize_aliases(original, patched)

    assert unrelated.stratified_cell_split is original
    assert package_child.stratified_cell_split is patched
