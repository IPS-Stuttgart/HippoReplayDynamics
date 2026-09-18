from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from scripts.forecast_tirole_content_coverage import forecast_event
from scripts.verify_tirole_future_forecasts import reference_null, reference_scores


def test_dense_null_matches_independent_mode_flow_construction():
    rng = np.random.default_rng(71)
    transition = rng.uniform(0.05, 1, (7, 7))
    transition /= transition.sum(axis=1, keepdims=True)
    op = NeuralOperator(np.ones(7) / 7, transition, np.ones(7) / 7)
    null = OccupancyMatchedNull.from_operator(op)
    dense = reference_null(transition)
    np.testing.assert_allclose(dense, null.step(np.eye(7)), atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.diag(dense), np.diag(transition), atol=1e-12, rtol=0)


def test_direct_forecast_reconstructs_every_baseline():
    rng = np.random.default_rng(15)
    transition = rng.uniform(0.01, 1, (5, 5))
    transition /= transition.sum(axis=1, keepdims=True)
    emissions = rng.uniform(0.01, 1, (8, 5))
    emissions /= emissions.sum(axis=0, keepdims=True)
    params = {"initial": np.ones(5) / 5, "transition": transition, "occupancy": np.ones(5) / 5, "probabilities": emissions, "global_probability": np.ones(8) / 8}
    fit = SimpleNamespace(**params)
    op = NeuralOperator(fit.initial, transition, fit.occupancy)
    matched = OccupancyMatchedNull.from_operator(op)
    dense = reference_null(transition)
    c = rng.poisson(0.5, (14, 8))
    train = np.array([0, 1, 2, 3, 4])
    held = np.array([5, 6, 7])
    _, scores = forecast_event(c, train, held, fit, op, matched)
    for row in scores:
        direct = reference_scores(c, train, held, params, dense, row["horizon"])
        for model, value in direct.items():
            assert value == pytest.approx(row["score_" + model], abs=1e-9)
