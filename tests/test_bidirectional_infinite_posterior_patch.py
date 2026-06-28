from __future__ import annotations

import numpy as np

from hipporeplayimm.bidirectional_infinite_evidence_patch import (
    _equal_prior_logp_and_weights,
    _safe_mixture_log_posterior,
)


def test_safe_bidirectional_posterior_mixture_keeps_all_impossible_terminal_non_nan() -> None:
    logp, weights = _equal_prior_logp_and_weights([-np.inf, -np.inf])

    mixed = _safe_mixture_log_posterior(
        [
            np.array([-np.inf, -np.inf], dtype=float),
            np.array([-np.inf, -np.inf], dtype=float),
        ],
        weights,
    )

    assert logp == -np.inf
    np.testing.assert_allclose(weights, np.array([0.5, 0.5], dtype=float))
    assert mixed is not None
    assert not np.isnan(mixed).any()
    assert np.isneginf(mixed).all()


def test_safe_bidirectional_posterior_mixture_preserves_impossible_trajectory_rows() -> None:
    _, weights = _equal_prior_logp_and_weights([-np.inf, -np.inf])

    mixed = _safe_mixture_log_posterior(
        [
            np.array([[0.0, -np.inf], [-np.inf, -np.inf]], dtype=float),
            np.array([[-np.inf, 0.0], [-np.inf, -np.inf]], dtype=float),
        ],
        weights,
    )

    assert mixed is not None
    assert not np.isnan(mixed).any()
    np.testing.assert_allclose(np.exp(mixed[0]).sum(), 1.0)
    assert np.isneginf(mixed[1]).all()
