from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import _poisson_log_emissions


@pytest.mark.parametrize("overdispersion", [0.0, 0.5], ids=["poisson", "negative-binomial"])
@pytest.mark.parametrize("dt", [1.0, np.array([1.0])], ids=["scalar-duration", "vector-duration"])
def test_positive_count_at_zero_rate_has_impossible_likelihood(
    overdispersion: float,
    dt: float | np.ndarray,
) -> None:
    log_likelihood = _poisson_log_emissions(
        np.array([[1]], dtype=int),
        np.array([[0.0, 2.0]], dtype=float),
        dt,
        negative_binomial_overdispersion=overdispersion,
    )

    assert np.isneginf(log_likelihood[0, 0])
    assert np.isfinite(log_likelihood[0, 1])


@pytest.mark.parametrize("overdispersion", [0.0, 0.5], ids=["poisson", "negative-binomial"])
def test_zero_count_at_zero_rate_has_unit_probability(overdispersion: float) -> None:
    log_likelihood = _poisson_log_emissions(
        np.array([[0]], dtype=int),
        np.array([[0.0]], dtype=float),
        1.0,
        likelihood_temperature=2.0,
        negative_binomial_overdispersion=overdispersion,
    )

    assert log_likelihood[0, 0] == 0.0


@pytest.mark.parametrize("overdispersion", [0.0, 0.5], ids=["poisson", "negative-binomial"])
def test_zero_weight_cell_does_not_make_zero_rate_support_impossible(
    overdispersion: float,
) -> None:
    actual = _poisson_log_emissions(
        np.array([[1, 0]], dtype=int),
        np.array([[0.0], [2.0]], dtype=float),
        1.0,
        cell_weights=(weight for weight in (0.0, 1.0)),
        negative_binomial_overdispersion=overdispersion,
    )
    expected = _poisson_log_emissions(
        np.array([[0]], dtype=int),
        np.array([[2.0]], dtype=float),
        1.0,
        negative_binomial_overdispersion=overdispersion,
    )

    np.testing.assert_allclose(actual, expected)
