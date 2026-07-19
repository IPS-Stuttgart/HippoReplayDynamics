from __future__ import annotations

import numpy as np

from hipporeplayimm.state_space_utils import (
    _full_grid_normalized_pairwise_gaussian_log_prob,
)


def test_pairwise_gaussian_keeps_nearest_support_when_all_scaled_distances_overflow() -> None:
    predicted = np.array([[0.0]], dtype=float)
    support = np.array([[2.0], [3.0]], dtype=float)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        log_prob = _full_grid_normalized_pairwise_gaussian_log_prob(
            predicted,
            support,
            support,
            1.0e-308,
        )

    assert log_prob[0, 0] == 0.0
    assert np.isneginf(log_prob[0, 1])
    np.testing.assert_array_equal(np.exp(log_prob), np.array([[1.0, 0.0]]))


def test_pairwise_gaussian_overflow_fallback_respects_valid_support_mask() -> None:
    predicted = np.array([[0.0]], dtype=float)
    all_support = np.array([[1.0], [2.0], [3.0]], dtype=float)
    observed = all_support[1:]

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        log_prob = _full_grid_normalized_pairwise_gaussian_log_prob(
            predicted,
            observed,
            all_support,
            1.0e-308,
            valid_bin_mask=np.array([False, True, True]),
        )

    np.testing.assert_array_equal(np.exp(log_prob), np.array([[1.0, 0.0]]))
