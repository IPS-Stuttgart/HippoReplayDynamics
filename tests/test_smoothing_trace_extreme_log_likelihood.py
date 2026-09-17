from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from hipporeplayimm.smoothing_trace import first_order_smoothing_trace
from hipporeplayimm.state_space import _forward_backward_first_order


def test_smoothing_trace_preserves_extreme_finite_log_likelihood_contrasts() -> None:
    """Finite log odds below exp() range must not be clipped into extra mass."""

    log_likelihood = np.array(
        [
            [0.0, -1000.0],
            [-1000.0, 0.0],
        ],
        dtype=float,
    )
    transition = csr_matrix(np.eye(2, dtype=float))

    expected_log_evidence, expected_log_smoothed = _forward_backward_first_order(
        log_likelihood,
        transition,
    )
    trace = first_order_smoothing_trace(log_likelihood, transition)

    np.testing.assert_allclose(expected_log_evidence, -1000.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        trace.log_evidence,
        expected_log_evidence,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace.log_predictive_probabilities.sum(),
        expected_log_evidence,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace.smoothed_probabilities,
        np.exp(expected_log_smoothed),
        rtol=0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        trace.smoothed_probabilities,
        np.full((2, 2), 0.5, dtype=float),
        rtol=0.0,
        atol=1e-13,
    )
    assert np.all(np.isfinite(trace.log_predictive_probabilities))
