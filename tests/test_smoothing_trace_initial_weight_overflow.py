from __future__ import annotations

import numpy as np

from hipporeplayimm.smoothing_trace import first_order_smoothing_trace


def test_large_finite_initial_weights_are_scale_invariant() -> None:
    log_likelihood = np.zeros((1, 2), dtype=float)
    transition = np.eye(2, dtype=float)

    baseline = first_order_smoothing_trace(
        log_likelihood,
        transition,
        initial_probabilities=np.array([1.0, 1.0], dtype=float),
    )
    scaled = first_order_smoothing_trace(
        log_likelihood,
        transition,
        initial_probabilities=np.array([1e308, 1e308], dtype=float),
    )

    np.testing.assert_array_equal(
        scaled.predicted_probabilities,
        baseline.predicted_probabilities,
    )
    np.testing.assert_array_equal(
        scaled.filtered_probabilities,
        baseline.filtered_probabilities,
    )
    np.testing.assert_array_equal(
        scaled.smoothed_probabilities,
        baseline.smoothed_probabilities,
    )
