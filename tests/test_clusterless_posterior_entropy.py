import numpy as np

from hipporeplayimm.clusterless_position_validation import _posterior_entropy


def test_posterior_entropy_ignores_zero_probability_bins() -> None:
    posterior = np.array([1.0, 0.0, 0.0])
    log_posterior = np.array([0.0, -np.inf, -np.inf])

    entropy = _posterior_entropy(posterior, log_posterior)

    assert np.isfinite(entropy)
    assert entropy == 0.0


def test_posterior_entropy_matches_binary_entropy() -> None:
    posterior = np.array([0.25, 0.75, 0.0])
    log_posterior = np.log(posterior, where=posterior > 0.0, out=np.full_like(posterior, -np.inf))

    entropy = _posterior_entropy(posterior, log_posterior)

    expected = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    assert np.isclose(entropy, expected)
