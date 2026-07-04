from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.result_improvements import stratified_cell_split


def test_stratified_cell_split_rejects_invalid_n_strata() -> None:
    cells = np.arange(8)
    scores = np.linspace(0.0, 1.0, cells.size)

    for invalid in (True, 0, -1, 1.5, np.nan):
        with pytest.raises(ValueError, match="n_strata must be a positive integer"):
            stratified_cell_split(cells, scores, 0.25, 1, n_strata=invalid)


def test_stratified_cell_split_accepts_integral_float_n_strata() -> None:
    cells = np.arange(8)
    scores = np.linspace(0.0, 1.0, cells.size)

    train, test = stratified_cell_split(cells, scores, 0.25, 1, n_strata=4.0)

    assert train.size + test.size == cells.size
    assert test.size == 2
    assert np.intersect1d(train, test).size == 0
