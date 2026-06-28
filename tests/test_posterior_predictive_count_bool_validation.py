from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.advanced_result_diagnostics import (
    posterior_predictive_count_checks,
    posterior_predictive_poisson_log_score,
)


def test_posterior_predictive_count_checks_reject_boolean_matrices() -> None:
    observed = np.array([[0.0, 1.0], [2.0, 0.0]])
    expected = np.array([[0.2, 0.8], [1.5, 0.1]])
    boolean_counts = np.array([[True, False], [False, True]])

    invalid_inputs = [
        {"observed_counts": boolean_counts, "expected_counts": expected},
        {"observed_counts": observed, "expected_counts": boolean_counts},
        {"observed_counts": observed, "expected_counts": expected, "variance_counts": boolean_counts},
    ]
    for kwargs in invalid_inputs:
        with pytest.raises(ValueError, match="not booleans"):
            posterior_predictive_count_checks(**kwargs)


def test_posterior_predictive_poisson_log_score_rejects_boolean_matrices() -> None:
    observed = np.array([[0.0, 1.0], [2.0, 0.0]])
    expected = np.array([[0.2, 0.8], [1.5, 0.1]])
    boolean_counts = np.array([[True, False], [False, True]])

    with pytest.raises(ValueError, match="not booleans"):
        posterior_predictive_poisson_log_score(boolean_counts, expected)
    with pytest.raises(ValueError, match="not booleans"):
        posterior_predictive_poisson_log_score(observed, boolean_counts)
