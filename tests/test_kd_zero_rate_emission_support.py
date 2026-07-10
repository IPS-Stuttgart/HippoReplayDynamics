from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.kd_reference import poisson_log_emissions


@pytest.mark.parametrize(
    "dt",
    [1.0, np.array([1.0])],
    ids=["scalar-duration", "vector-duration"],
)
def test_kd_positive_count_at_zero_rate_has_impossible_likelihood(
    dt: float | np.ndarray,
) -> None:
    log_likelihood = poisson_log_emissions(
        np.array([[1]], dtype=int),
        np.array([[0.0, 2.0]], dtype=float),
        dt,
    )

    assert np.isneginf(log_likelihood[0, 0])
    assert np.isfinite(log_likelihood[0, 1])


@pytest.mark.parametrize(
    "dt",
    [1.0, np.array([1.0])],
    ids=["scalar-duration", "vector-duration"],
)
def test_kd_zero_count_at_zero_rate_has_unit_probability(
    dt: float | np.ndarray,
) -> None:
    log_likelihood = poisson_log_emissions(
        np.array([[0]], dtype=int),
        np.array([[0.0]], dtype=float),
        dt,
    )

    assert log_likelihood[0, 0] == 0.0
