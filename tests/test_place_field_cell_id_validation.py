from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.advanced_result_diagnostics import place_field_quality_from_arrays


def test_place_field_quality_rejects_invalid_cell_ids():
    rates = np.array([[1.0, 10.0, 1.0], [0.5, 0.5, 0.5]])
    occupancy = np.array([1.0, 1.0, 1.0])

    for cell_ids in ([11.5, 12], [True, 12], [np.nan, 12], [[11], [12]]):
        with pytest.raises(ValueError, match="cell_ids"):
            place_field_quality_from_arrays(rates, occupancy, cell_ids=cell_ids)


def test_place_field_quality_accepts_numpy_integer_cell_ids():
    rates = np.array([[1.0, 10.0, 1.0], [0.5, 0.5, 0.5]])
    occupancy = np.array([1.0, 1.0, 1.0])

    quality = place_field_quality_from_arrays(
        rates,
        occupancy,
        cell_ids=np.array([np.int64(21), np.int64(22)]),
    )

    assert quality["cell_id"].tolist() == [21, 22]
