from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.encoding import _poisson_log_emissions


def _nested_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def test_poisson_log_emissions_rejects_boolean_spike_counts() -> None:
    with pytest.raises(ValueError, match="spike_counts.*boolean"):
        _poisson_log_emissions(
            np.array([[True, False]], dtype=bool),
            np.ones((2, 3), dtype=float),
            0.02,
        )


def test_poisson_log_emissions_rejects_boolean_rates() -> None:
    with pytest.raises(ValueError, match="rates_hz.*boolean"):
        _poisson_log_emissions(
            np.array([[0, 1]], dtype=int),
            np.array(
                [
                    [True, False, True],
                    [False, True, False],
                ],
                dtype=bool,
            ),
            0.02,
        )


def test_poisson_log_emissions_rejects_object_boolean_inputs() -> None:
    with pytest.raises(ValueError, match="spike_counts.*boolean"):
        _poisson_log_emissions(
            np.array([[True, 0]], dtype=object),
            np.ones((2, 3), dtype=float),
            0.02,
        )


def test_poisson_log_emissions_rejects_nested_object_boolean_counts() -> None:
    counts = np.array([[0, 1]], dtype=object)
    counts[0, 0] = _nested_scalar(True)

    with pytest.raises(ValueError, match="spike_counts.*boolean"):
        _poisson_log_emissions(
            counts,
            np.ones((2, 3), dtype=float),
            0.02,
        )


def test_poisson_log_emissions_rejects_nested_object_boolean_rates() -> None:
    rates = np.ones((2, 3), dtype=object)
    rates[0, 1] = _nested_scalar(np.bool_(True))

    with pytest.raises(ValueError, match="rates_hz.*boolean"):
        _poisson_log_emissions(
            np.array([[0, 1]], dtype=int),
            rates,
            0.02,
        )


@pytest.mark.parametrize(
    "count",
    [
        Decimal("9007199254740993.5"),
        Decimal("1.0000000000000000000000000000000001"),
        "2.5",
    ],
)
def test_poisson_log_emissions_rejects_fractional_counts_before_float_conversion(
    count: object,
) -> None:
    with pytest.raises(ValueError, match="spike_counts.*integer"):
        _poisson_log_emissions(
            np.array([[count]], dtype=object),
            np.ones((1, 2), dtype=float),
            0.02,
        )


def test_poisson_log_emissions_rejects_complex_counts() -> None:
    with pytest.raises(ValueError, match="spike_counts.*real integer"):
        _poisson_log_emissions(
            np.array([[1.0 + 0.0j]], dtype=complex),
            np.ones((1, 2), dtype=float),
            0.02,
        )


@pytest.mark.parametrize("count", [Decimal("2.0"), "2e0", b"2.0"])
def test_poisson_log_emissions_accepts_exact_object_count_forms(
    count: object,
) -> None:
    rates = np.array([[1.0, 3.0]], dtype=float)
    expected = _poisson_log_emissions(
        np.array([[2]], dtype=int),
        rates,
        0.02,
    )
    actual = _poisson_log_emissions(
        np.array([[count]], dtype=object),
        rates,
        0.02,
    )

    np.testing.assert_array_equal(actual, expected)