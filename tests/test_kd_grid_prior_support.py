import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.kd_reference import marginalize_grid_log_evidence


def test_grid_marginalization_preserves_exact_zero_prior_support():
    grid = np.array([[0.0, 1000.0], [-3.0, -2.0]])
    prior = np.array([1.0, 0.0])

    marginalized = marginalize_grid_log_evidence(grid, prior)

    np.testing.assert_allclose(marginalized, np.array([0.0, -3.0]))


@pytest.mark.parametrize("excluded_value", [np.nan, np.inf])
def test_grid_marginalization_ignores_nonfinite_values_outside_prior_support(excluded_value):
    grid = np.array([[0.0, excluded_value], [-3.0, excluded_value]])
    prior = np.array([1.0, 0.0])

    marginalized = marginalize_grid_log_evidence(grid, prior)

    np.testing.assert_allclose(marginalized, np.array([0.0, -3.0]))


@pytest.mark.parametrize("supported_value", [np.nan, np.inf])
def test_grid_marginalization_rejects_invalid_values_inside_prior_support(supported_value):
    grid = np.array([[0.0, supported_value], [-3.0, -2.0]])
    prior = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="positive prior mass"):
        marginalize_grid_log_evidence(grid, prior)


def test_grid_marginalization_preserves_impossible_supported_hypotheses():
    grid = np.array([[0.0, -np.inf], [-np.inf, -3.0]])
    prior = np.array([0.5, 0.5])

    marginalized = marginalize_grid_log_evidence(grid, prior)

    np.testing.assert_allclose(
        marginalized,
        np.array([np.log(0.5), -3.0 + np.log(0.5)]),
    )


def test_grid_marginalization_matches_logsumexp_for_positive_prior():
    grid = np.array([[-4.0, -2.0, -3.0], [-1.0, -5.0, -2.0]])
    prior = np.array([0.2, 0.3, 0.5])

    marginalized = marginalize_grid_log_evidence(grid, prior)

    expected = logsumexp(grid + np.log(prior), axis=1)
    np.testing.assert_allclose(marginalized, expected)


def test_grid_marginalization_normalizes_unnormalized_prior_weights():
    grid = np.array([[-4.0, -2.0, -3.0], [-1.0, -5.0, -2.0]])

    normalized = marginalize_grid_log_evidence(grid, np.array([0.2, 0.3, 0.5]))
    unnormalized = marginalize_grid_log_evidence(grid, np.array([2.0, 3.0, 5.0]))

    np.testing.assert_allclose(unnormalized, normalized)


@pytest.mark.parametrize(
    "prior, message",
    [
        (np.array([0.0, 0.0]), "positive mass"),
        (np.array([1.0, -0.1]), "negative"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, np.inf]), "finite"),
    ],
)
def test_grid_marginalization_rejects_invalid_priors(prior, message):
    with pytest.raises(ValueError, match=message):
        marginalize_grid_log_evidence(np.zeros((2, 2)), prior)
