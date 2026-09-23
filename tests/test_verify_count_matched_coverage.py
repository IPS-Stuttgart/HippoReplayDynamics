"""Cross-check independent dense calculations and causal forecast horizon."""

import numpy as np

from hipporeplayimm.independent_rejected_forecast import forecast_distributions, score_predictions
from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from scripts.verify_count_matched_coverage import null_transition, predicted_score, trajectory


def fixture():
    rng = np.random.default_rng(73)
    transition = rng.uniform(.1, 1, (7, 7)) + 5 * np.eye(7)
    transition /= transition.sum(axis=1, keepdims=True)
    initial = np.full(7, 1 / 7)
    rates = rng.uniform(.1, 3, (8, 7))
    counts = rng.poisson(1, (10, 8))
    return transition, initial, rates, counts


def test_independent_dense_null_reproduces_constraints_and_operator():
    transition, initial, _, _ = fixture()
    operator = NeuralOperator(initial, transition, initial)
    null = OccupancyMatchedNull.from_operator(operator)
    dense = null_transition(transition)
    np.testing.assert_allclose(dense, null.step(np.eye(7)), atol=1e-10)
    np.testing.assert_allclose(np.diag(dense), np.diag(transition), atol=1e-12)


def test_independent_predictive_score_matches_both_causal_filters():
    transition, initial, rates, counts = fixture()
    operator = NeuralOperator(initial, transition, initial)
    null = OccupancyMatchedNull.from_operator(operator)
    predictions = forecast_distributions(counts[:, :5], rates[:5], operator, null, (2,))
    scores = score_predictions(predictions, counts[:, 5:], rates[5:], np.ones(3))[0]
    for name, matrix in [("dynamic", transition), ("matched_own", null_transition(transition))]:
        got = predicted_score(counts[:, :5], rates[:5], counts[:, 5:], rates[5:], initial, matrix)
        np.testing.assert_allclose(got, scores["score_" + name], atol=1e-10)


def test_final_two_inference_bins_cannot_affect_forecast_score():
    transition, initial, rates, counts = fixture()
    first = predicted_score(counts[:, :5], rates[:5], counts[:, 5:], rates[5:], initial, transition)
    perturbed = counts[:, :5].copy()
    perturbed[-2:] = 1000
    second = predicted_score(perturbed, rates[:5], counts[:, 5:], rates[5:], initial, transition)
    assert first == second


def test_independent_geometry_tie_and_boundary():
    grid = np.column_stack([np.arange(20) * 5, np.zeros(20)])
    counts = np.full((10, 3), 1)
    assert trajectory(np.arange(10), grid, counts) == (True, 10, 45.0)
    assert trajectory(np.zeros(10, dtype=int), grid, counts) == (False, 10, 0.0)
    assert trajectory(np.arange(10), grid, np.zeros_like(counts)) == (False, 0, 0.0)
