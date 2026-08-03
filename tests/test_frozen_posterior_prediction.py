from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.frozen_posterior_prediction import (
    frozen_smoothed_marginal_log_score,
    normalized_log_posterior,
    posterior_sha256,
)


def test_frozen_score_matches_direct_mixture_calculation() -> None:
    posterior = np.log(np.asarray([[0.75, 0.25], [0.2, 0.8]], dtype=float))
    heldout = np.log(np.asarray([[0.4, 0.1], [0.3, 0.9]], dtype=float))

    result = frozen_smoothed_marginal_log_score(posterior, heldout)
    expected = np.asarray(
        [
            np.log(0.75 * 0.4 + 0.25 * 0.1),
            np.log(0.2 * 0.3 + 0.8 * 0.9),
        ]
    )

    np.testing.assert_allclose(result.per_time_log_score, expected)
    assert result.total_log_score == pytest.approx(float(expected.sum()))
    assert result.mean_log_score_per_time_bin == pytest.approx(float(expected.mean()))


def test_frozen_score_normalizes_posterior_rows() -> None:
    posterior = np.asarray([[1.0, 0.0], [-8.0, -7.0]], dtype=float)
    shifted = posterior + np.asarray([[100.0], [-200.0]])
    heldout = np.asarray([[-2.0, -3.0], [-4.0, -1.0]], dtype=float)

    first = frozen_smoothed_marginal_log_score(posterior, heldout)
    second = frozen_smoothed_marginal_log_score(shifted, heldout)

    np.testing.assert_allclose(first.per_time_log_score, second.per_time_log_score)
    assert first.posterior_sha256 == second.posterior_sha256


def test_heldout_values_cannot_change_frozen_posterior_hash() -> None:
    posterior = np.asarray([[0.0, -1.0], [-2.0, 0.0]], dtype=float)
    first = frozen_smoothed_marginal_log_score(
        posterior,
        np.asarray([[-1.0, -3.0], [-4.0, -1.0]]),
    )
    second = frozen_smoothed_marginal_log_score(
        posterior,
        np.asarray([[-9.0, -0.1], [-0.2, -8.0]]),
    )

    assert first.posterior_sha256 == second.posterior_sha256
    assert first.total_log_score != second.total_log_score


def test_normalized_posterior_and_hash_are_deterministic() -> None:
    values = np.asarray([[0.0, -2.0, -4.0], [-1.0, -1.0, -1.0]])
    normalized = normalized_log_posterior(values)
    np.testing.assert_allclose(logsumexp(normalized, axis=1), 0.0, atol=1e-14)
    assert posterior_sha256(values) == posterior_sha256(values.copy())


@pytest.mark.parametrize(
    ("posterior", "heldout", "message"),
    [
        (np.asarray([0.0, -1.0]), np.zeros((1, 2)), "shape"),
        (np.zeros((2, 2)), np.zeros((2, 3)), "match"),
        (np.asarray([[np.nan, 0.0]]), np.zeros((1, 2)), "NaN"),
        (np.zeros((1, 2)), np.asarray([[np.nan, 0.0]]), "NaN"),
    ],
)
def test_invalid_frozen_score_inputs_fail(
    posterior: np.ndarray,
    heldout: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        frozen_smoothed_marginal_log_score(posterior, heldout)
