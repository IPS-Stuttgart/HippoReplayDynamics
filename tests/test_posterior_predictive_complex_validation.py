from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.advanced_result_diagnostics import (
    posterior_predictive_count_checks,
    posterior_predictive_poisson_log_score,
)


def _nested_complex_table(value: object) -> np.ndarray:
    table = np.empty((1, 1), dtype=object)
    table[0, 0] = np.asarray(value, dtype=object)
    return table


@pytest.mark.parametrize(
    ("name", "observed", "expected", "variance"),
    [
        (
            "observed_counts",
            np.array([[np.complex128(1.0 + 2.0j)]]),
            np.array([[1.0]]),
            None,
        ),
        (
            "expected_counts",
            np.array([[1.0]]),
            np.array([[np.complex64(1.0 + 0.0j)]], dtype=object),
            None,
        ),
        (
            "variance_counts",
            np.array([[1.0]]),
            np.array([[1.0]]),
            _nested_complex_table(np.clongdouble(1.0 + 2.0j)),
        ),
    ],
)
def test_posterior_predictive_count_checks_reject_complex_inputs(
    name: str,
    observed: np.ndarray,
    expected: np.ndarray,
    variance: np.ndarray | None,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=rf"{name} must contain real numeric values"):
            posterior_predictive_count_checks(
                observed,
                expected,
                variance_counts=variance,
            )


def test_posterior_predictive_poisson_log_score_rejects_complex_inputs() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            ValueError,
            match="expected_counts must contain real numeric values",
        ):
            posterior_predictive_poisson_log_score(
                np.array([[1.0]]),
                _nested_complex_table(np.complex128(1.0 + 0.0j)),
            )
