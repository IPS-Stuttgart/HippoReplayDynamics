from __future__ import annotations

import numpy as np

from hipporeplayimm.reverse_models import _mixture_log_posterior


def test_mixture_log_posterior_excludes_zero_weight_component() -> None:
    forward = np.array(
        [
            [0.0, -np.inf],
            [0.0, -np.inf],
        ]
    )
    reverse = np.array(
        [
            [-np.inf, 0.0],
            [-np.inf, 0.0],
        ]
    )

    mixed = _mixture_log_posterior(
        forward,
        reverse,
        np.array([1.0, 0.0]),
    )

    assert mixed is not None
    np.testing.assert_array_equal(mixed, forward)


def test_mixture_log_posterior_ignores_only_available_zero_weight_posterior() -> None:
    reverse = np.array([[-np.inf, 0.0]])

    mixed = _mixture_log_posterior(
        None,
        reverse,
        np.array([1.0, 0.0]),
    )

    assert mixed is None
