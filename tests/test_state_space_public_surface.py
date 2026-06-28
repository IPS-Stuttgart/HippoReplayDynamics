from __future__ import annotations

import numpy as np

from hipporeplayimm.state_space import (
    _forward_backward_first_order_time_varying,
    _gaussian_transition_matrix,
)


def test_public_state_space_exports_time_varying_forward_backward() -> None:
    log_likelihood = np.array(
        [
            [0.0, -1.0, -2.0],
            [-1.0, 0.0, -1.5],
            [-2.0, -0.5, 0.0],
        ],
        dtype=float,
    )
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    transitions = [
        _gaussian_transition_matrix(bin_centers, 0.5, 6.0),
        _gaussian_transition_matrix(bin_centers, 1.0, 6.0),
    ]

    logp, posterior = _forward_backward_first_order_time_varying(
        log_likelihood,
        transitions,
    )

    assert np.isfinite(logp)
    assert posterior.shape == log_likelihood.shape
    np.testing.assert_allclose(
        np.exp(posterior).sum(axis=1),
        np.ones(log_likelihood.shape[0]),
        rtol=1e-12,
        atol=1e-12,
    )
