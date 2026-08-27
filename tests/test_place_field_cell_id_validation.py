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


def test_place_field_quality_preserves_extended_precision_integer_cell_ids():
    expected_cell_id = 2**53 + 1
    cell_id = np.longdouble(2**53) + np.longdouble(1)
    if int(cell_id) != expected_cell_id:
        pytest.skip("np.longdouble does not exceed float64 integer precision on this platform")

    quality = place_field_quality_from_arrays(
        np.array([[1.0, 10.0, 1.0]]),
        np.array([1.0, 1.0, 1.0]),
        cell_ids=[cell_id],
    )

    assert quality["cell_id"].tolist() == [expected_cell_id]


@pytest.mark.parametrize(
    ("rates_hz", "occupancy_s", "message"),
    [
        (np.array([[1.0, -2.0, 3.0]]), np.ones(3), "rates_hz"),
        (np.array([[1.0, np.nan, 3.0]]), np.ones(3), "rates_hz"),
        (np.array([[1.0, np.inf, 3.0]]), np.ones(3), "rates_hz"),
        (np.array([[1.0 + 0.0j, 2.0, 3.0]]), np.ones(3), "rates_hz"),
        (np.array([[1.0, 2.0, 3.0]]), np.array([1.0, -0.5, 1.0]), "occupancy_s"),
        (np.array([[1.0, 2.0, 3.0]]), np.array([1.0, np.nan, 1.0]), "occupancy_s"),
        (np.array([[1.0, 2.0, 3.0]]), np.array([1.0, np.inf, 1.0]), "occupancy_s"),
        (np.array([[1.0, 2.0, 3.0]]), np.array([1.0 + 0.0j, 1.0, 1.0]), "occupancy_s"),
    ],
)
def test_place_field_quality_rejects_invalid_numeric_inputs(
    rates_hz,
    occupancy_s,
    message,
):
    with pytest.raises(ValueError, match=message):
        place_field_quality_from_arrays(rates_hz, occupancy_s)


def test_place_field_quality_rejects_nested_complex_numeric_inputs():
    rates = np.empty((1, 3), dtype=object)
    rates[0] = [np.array(1.0 + 0.0j), 2.0, 3.0]

    with pytest.raises(ValueError, match="rates_hz"):
        place_field_quality_from_arrays(rates, np.ones(3))


def test_place_field_quality_preserves_zero_rate_and_zero_occupancy_support():
    quality = place_field_quality_from_arrays(
        np.array([[0.0, 1.0, 0.0]]),
        np.zeros(3),
        cell_ids=[7],
    )

    assert quality["cell_id"].tolist() == [7]
    assert np.isfinite(quality.loc[0, "spatial_information_bits_per_spike"])
