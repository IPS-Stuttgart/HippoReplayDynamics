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

    with pytest.raises(ValueError, match="observed_counts must contain integer count values"):
        posterior_predictive_count_checks(np.array([[0.5, 1.0]]), expected)

    with pytest.raises(ValueError, match="expected_counts must contain finite nonnegative values"):
        posterior_predictive_count_checks(observed, np.array([[0.2, -0.8]]))

    with pytest.raises(ValueError, match="variance_counts must contain finite nonnegative values"):
        posterior_predictive_count_checks(observed, expected, variance_counts=np.array([[0.2, np.nan]]))


def test_posterior_predictive_checks_reject_empty_count_tables():
    message = "must contain at least one time bin and one cell"

    with pytest.raises(ValueError, match=f"observed_counts {message}"):
        posterior_predictive_count_checks(np.empty((0, 2)), np.empty((0, 2)))

    with pytest.raises(ValueError, match=f"observed_counts {message}"):
        posterior_predictive_count_checks(np.empty((1, 0)), np.empty((1, 0)))

    with pytest.raises(ValueError, match=f"expected_counts {message}"):
        posterior_predictive_count_checks(np.array([[0.0]]), np.empty((0, 1)))

    with pytest.raises(ValueError, match=f"variance_counts {message}"):
        posterior_predictive_count_checks(np.array([[0.0]]), np.array([[0.1]]), variance_counts=np.empty((0, 1)))

    with pytest.raises(ValueError, match=f"observed_counts {message}"):
        posterior_predictive_poisson_log_score(np.empty((0, 2)), np.empty((0, 2)))


def test_posterior_predictive_checks_allow_fractional_expectations_and_variances():
    observed = np.array([[0.0, 1.0]])
    expected = np.array([[0.2, 0.8]])
    variance = np.array([[0.3, 1.2]])

    checks = posterior_predictive_count_checks(observed, expected, variance_counts=variance)
    assert checks["predictive_check"].tolist() == [
        "total_spike_count",
        "silent_bin_fraction",
        "mean_abs_cell_z",
        "max_abs_cell_z",
    ]
    assert np.isfinite(posterior_predictive_poisson_log_score(observed, expected))


def test_posterior_predictive_poisson_log_score_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="expected_counts must contain finite nonnegative values"):
        posterior_predictive_poisson_log_score(np.array([[1.0]]), np.array([[-1.0]]))

    with pytest.raises(ValueError, match="observed_counts must contain finite nonnegative values"):
        posterior_predictive_poisson_log_score(np.array([[np.inf]]), np.array([[1.0]]))

    with pytest.raises(ValueError, match="observed_counts must contain integer count values"):
        posterior_predictive_poisson_log_score(np.array([[0.5]]), np.array([[1.0]]))


def test_posterior_predictive_poisson_log_score_preserves_zero_mean_support():
    assert posterior_predictive_poisson_log_score(np.array([[0.0]]), np.array([[0.0]])) == 0.0
    assert np.isneginf(
        posterior_predictive_poisson_log_score(np.array([[1.0]]), np.array([[0.0]]))
    )
