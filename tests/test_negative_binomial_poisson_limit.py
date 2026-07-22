from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import _poisson_log_emissions


_TINY_POSITIVE_OVERDISPERSIONS = (
    1.0e-12,
    np.nextafter(0.0, 1.0),
)


@pytest.mark.parametrize(
    "overdispersion",
    _TINY_POSITIVE_OVERDISPERSIONS,
)
def test_tiny_positive_overdispersion_converges_to_poisson(
    overdispersion: float,
) -> None:
    counts = np.array([[0], [1], [3]], dtype=int)
    rates_hz = np.array([[2.0, 4.0]], dtype=float)

    actual = _poisson_log_emissions(
        counts,
        rates_hz,
        1.0,
        negative_binomial_overdispersion=overdispersion,
    )
    poisson = _poisson_log_emissions(
        counts,
        rates_hz,
        1.0,
        negative_binomial_overdispersion=0.0,
    )

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, poisson, rtol=0.0, atol=1.0e-10)


@pytest.mark.parametrize(
    "overdispersion",
    _TINY_POSITIVE_OVERDISPERSIONS,
)
def test_tiny_overdispersion_preserves_exact_zero_rate_support(
    overdispersion: float,
) -> None:
    counts = np.array([[0], [1]], dtype=int)
    rates_hz = np.array([[0.0, 2.0]], dtype=float)

    actual = _poisson_log_emissions(
        counts,
        rates_hz,
        1.0,
        negative_binomial_overdispersion=overdispersion,
    )
    poisson = _poisson_log_emissions(
        counts,
        rates_hz,
        1.0,
        negative_binomial_overdispersion=0.0,
    )

    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(poisson))
    finite = np.isfinite(poisson)
    np.testing.assert_allclose(
        actual[finite],
        poisson[finite],
        rtol=0.0,
        atol=1.0e-10,
    )
