"""Numerical precision must not change nuisance likelihood or penalties."""
import numpy as np
import pytest
from scipy import sparse
import json
from pathlib import Path

from scripts._run_pair_rate_glm import precise_scaled_solution
from scripts.verify_run_pair_rate_glm_stress import poisson_newton_refinement


def test_newton_recovers_analytic_intercept_and_zero_penalized_features():
    x = sparse.csr_matrix(np.zeros((20, 3)))
    y = np.r_[np.zeros(17), 1, 2, 1]
    weight = np.full(20, .05)
    answer = poisson_newton_refinement(np.array([.01, -.02, .03, -1.5]), x, y, weight, np.array([.1, .1, .01]))
    np.testing.assert_allclose(answer[:-1], 0, atol=1e-12)
    np.testing.assert_allclose(answer[-1], np.log(.2), atol=1e-12)


def test_grouped_penalty_solution_is_invariant_to_feature_parameterization():
    rng = np.random.default_rng(71)
    x = sparse.csr_matrix(rng.normal(size=(100, 4)))
    y = rng.poisson(.1 * np.exp(np.asarray(x @ np.array([.1, .1, .2, -.1])).ravel()))
    weight = np.full(100, .01)
    penalty = np.array([1., 1., .01, .01])
    initial = np.r_[np.zeros(4), np.log(weight @ y)]
    unscaled = poisson_newton_refinement(initial, x, y, weight, penalty)
    scale = np.array([.1, .1, 1., 1.])
    scaled = poisson_newton_refinement(initial, x.multiply(scale).tocsr(), y, weight, .01)
    np.testing.assert_allclose(unscaled[:-1], scaled[:-1] * scale, atol=1e-11)
    np.testing.assert_allclose(unscaled[-1], scaled[-1], atol=1e-11)
    producer = precise_scaled_solution(initial, x.multiply(scale).tocsr(), y, weight, .01)
    np.testing.assert_allclose(producer, scaled, atol=1e-11)


def test_nonfinite_start_is_rejected():
    with pytest.raises(ValueError, match="Nonfinite Newton"):
        poisson_newton_refinement(np.array([np.nan, -2.]), sparse.csr_matrix(np.ones((5, 1))),
                                 np.ones(5), np.full(5, .2), .1)


def test_precision_protocol_does_not_change_statistical_model_or_support():
    root = Path(__file__).parents[1] / "docs"
    original = json.loads((root / "run_pair_coordination_endpoint_v5_support_protocol.json").read_text())
    refined = json.loads((root / "run_pair_coordination_endpoint_v6_support_precision_protocol.json").read_text())
    descriptions = {"protocol_id", "frozen_before", "scope", "rate_model", "claim_boundary", "glm_group_solver"}
    assert set(original) == set(refined)
    assert all(refined[k] == original[k] for k in set(original) - descriptions)
    assert refined["glm_group_solver"] == "scaled_scipy_lbfgs_newton"
