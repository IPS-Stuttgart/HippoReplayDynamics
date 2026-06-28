from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.advanced_result_diagnostics import (
    posterior_predictive_count_checks,
    posterior_predictive_poisson_log_score,
)


def test_posterior_predictive_count_checks_reject_impossible_count_inputs():
    observed = np.array([[0.0, 1.0]])
    expected = np.array([[0.2, 0.8]])

    with pytest.raises(ValueError, match="observed_counts must contain finite nonnegative values"):
        posterior_predictive_count_checks(np.array([[0.0, -1.0]]), expected)

    with pytest.raises(ValueError, match="expected_counts must contain finite nonnegative values"):
        posterior_predictive_count_checks(observed, np.array([[0.2, -0.8]]))

    with pytest.raises(ValueError, match="variance_counts must contain finite nonnegative values"):
        posterior_predictive_count_checks(observed, expected, variance_counts=np.array([[0.2, np.nan]]))


def test_posterior_predictive_poisson_log_score_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="expected_counts must contain finite nonnegative values"):
        posterior_predictive_poisson_log_score(np.array([[1.0]]), np.array([[-1.0]]))

    with pytest.raises(ValueError, match="observed_counts must contain finite nonnegative values"):
        posterior_predictive_poisson_log_score(np.array([[np.inf]]), np.array([[1.0]]))
