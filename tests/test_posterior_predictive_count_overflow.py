from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.advanced_result_diagnostics import posterior_predictive_count_checks


@pytest.mark.parametrize(
    ("observed", "variance"),
    [
        (
            np.full((1, 2), np.finfo(float).max, dtype=float),
            np.full((1, 2), np.finfo(float).max, dtype=float),
        ),
        (
            np.array([[np.finfo(float).max]], dtype=float),
            np.zeros((1, 1), dtype=float),
        ),
    ],
    ids=["aggregate-overflow", "standardized-residual-overflow"],
)
def test_posterior_predictive_count_checks_rejects_derived_overflow_without_warning(
    observed: np.ndarray,
    variance: np.ndarray,
) -> None:
    expected = np.zeros_like(observed)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(
            ValueError,
            match="posterior-predictive diagnostics exceed floating-point range",
        ):
            posterior_predictive_count_checks(
                observed,
                expected,
                variance_counts=variance,
            )


def test_posterior_predictive_count_checks_keeps_finite_diagnostics() -> None:
    checks = posterior_predictive_count_checks(
        np.array([[0, 2]], dtype=int),
        np.array([[0.5, 1.5]], dtype=float),
        variance_counts=np.ones((1, 2), dtype=float),
    )

    total = checks.loc[checks["predictive_check"].eq("total_spike_count")].iloc[0]
    assert total["observed"] == 2.0
    assert total["expected"] == 2.0
    assert total["z_score"] == 0.0
