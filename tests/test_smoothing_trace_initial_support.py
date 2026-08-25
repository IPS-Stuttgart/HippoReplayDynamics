from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.smoothing_trace import first_order_smoothing_trace


def test_initial_support_validation_is_scale_invariant() -> None:
    valid = np.array([True, False])
    transition = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="zero mass outside valid_bin_mask"):
        first_order_smoothing_trace(
            np.zeros((1, 2), dtype=float),
            transition,
            initial_probabilities=np.array([1e-15, 1e-15], dtype=float),
            valid_bin_mask=valid,
        )


def test_tiny_valid_initial_distribution_is_normalized_on_valid_support() -> None:
    valid = np.array([True, False])
    transition = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=float)

    trace = first_order_smoothing_trace(
        np.zeros((1, 2), dtype=float),
        transition,
        initial_probabilities=np.array([1e-15, 0.0], dtype=float),
        valid_bin_mask=valid,
    )

    np.testing.assert_array_equal(
        trace.predicted_probabilities[0],
        np.array([1.0, 0.0]),
    )
