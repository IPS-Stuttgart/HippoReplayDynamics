import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from hipporeplayimm.regional_grid_mixture import (
    em_update,
    fit_grid,
    grid_likelihood,
    normalize_likelihood,
    smoothing_matrix,
)


def test_first_iterate_equals_mean_flat_posterior():
    ll = np.array([[-2.0, -1.0, -4.0], [-1.0, -5.0, -2.0]])
    likelihood = normalize_likelihood(ll)
    first = em_update(likelihood, np.ones(3) / 3)
    expected = np.exp(ll - logsumexp(ll, axis=1, keepdims=True)).mean(axis=0)
    np.testing.assert_allclose(first, expected)


def test_known_identifiable_mixture_and_separate_optimizer():
    emissions = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    ll = np.log(np.repeat(emissions, [45, 31, 24], axis=0))
    smooth = smoothing_matrix(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    for penalty in (0.0, 0.01, 0.1):
        fit = fit_grid(ll, smooth, penalty)
        assert fit["converged"]
        reference = minimize(
            lambda p, penalty=penalty: -logsumexp(ll + np.log(np.maximum(p, 1e-300)), axis=1).mean() + penalty * 0.5 * p @ (smooth @ p),
            np.ones(3) / 3,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 3,
            constraints={"type": "eq", "fun": lambda p: p.sum() - 1},
            options={"ftol": 1e-13, "maxiter": 1000},
        )
        assert reference.success
        np.testing.assert_allclose(fit["pi"], reference.x, atol=1e-5)
        if penalty == 0:
            np.testing.assert_allclose(fit["pi"], (np.array([0.45, 0.31, 0.24]) - 0.1) / 0.7, atol=1e-5)


def test_flat_likelihood_is_flagged_not_evidence_for_area_mass():
    fit = fit_grid(np.zeros((10, 3)), csr_matrix((3, 3)))
    assert fit["flat_likelihood"] and fit["converged"]
    np.testing.assert_allclose(fit["pi"], 1 / 3)


def test_conditional_counts_silence_and_censoring():
    rates = np.array([[2.0, 8.0], [8.0, 2.0]])
    counts = np.array([[1, 0], [0, 0]])
    conditional = grid_likelihood(counts, rates, "conditional_multinomial")
    np.testing.assert_allclose(conditional[0], np.log([0.2, 0.8]))
    np.testing.assert_allclose(conditional[1], 0)
    censored = grid_likelihood(counts, rates, "fixed_total_censored", np.array([2, 1]), np.array([20.0, 40.0]))
    np.testing.assert_allclose(censored[0], np.log([2 / 20 * 10 / 20, 8 / 40 * 30 / 40]))
    np.testing.assert_allclose(censored[1], np.log([10 / 20, 30 / 40]))


def test_subset_count_can_carry_location_information_missing_from_conditional():
    rates = np.array([[1.0, 9.0]])
    counts = np.array([[1], [0]])
    np.testing.assert_allclose(grid_likelihood(counts, rates, "conditional_multinomial"), 0)
    ll = grid_likelihood(counts, rates, "fixed_total_censored", np.ones(2), np.full(2, 10.0))
    assert ll[0, 1] > ll[0, 0] and ll[1, 0] > ll[1, 1]


def test_smoothing_is_nonnegative_and_ignores_disconnected_gap():
    smooth = smoothing_matrix(np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]]))
    np.testing.assert_allclose(smooth @ np.ones(4), 0)
    assert smooth[1, 2] == 0
    p = np.array([0.1, 0.2, 0.3, 0.4])
    assert p @ (smooth @ p) >= 0


def test_invalid_likelihood_or_subset_rejected():
    with pytest.raises(ValueError):
        fit_grid(np.full((2, 3), np.nan), csr_matrix((3, 3)))
    with pytest.raises(ValueError):
        grid_likelihood(np.array([[2]]), np.array([[1.0, 1.0]]), "fixed_total_censored", [1], [2.0, 2.0])
