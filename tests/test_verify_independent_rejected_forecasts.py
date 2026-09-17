"""Checks for the separately implemented result verifier."""

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.independent_rejected_forecast import (
    classify,
    forecast_distributions,
    score_predictions,
)
from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from scripts.verify_independent_rejected_forecasts import ref_geometry, ref_null, ref_scores


def test_dense_reference_null_matches_constraints_and_production():
    rng = np.random.default_rng(17)
    a = rng.random((7, 7)) + 0.01
    a /= a.sum(axis=1, keepdims=True)
    op = NeuralOperator(np.full(7, 1 / 7), a, np.full(7, 1 / 7))
    actual = OccupancyMatchedNull.from_operator(op).step(np.eye(7))
    expected = ref_null(a)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_reference_prediction_reconstructs_every_baseline():
    rng = np.random.default_rng(48)
    a = rng.random((5, 5)) + 0.02
    a /= a.sum(axis=1, keepdims=True)
    p = rng.random((8, 5)) + 0.1
    p /= p.sum(axis=0)
    fit = dict(transition=a, initial=np.full(5, 0.2), probabilities=p, global_probability=np.full(8, 1 / 8))
    counts = rng.poisson(0.8, (11, 8))
    tr, held = np.arange(5), np.arange(5, 8)
    op = NeuralOperator(fit["initial"], a, np.full(5, 0.2))
    null = OccupancyMatchedNull.from_operator(op)
    scores = score_predictions(forecast_distributions(counts[:, tr], p[tr], op, null), counts[:, held], p[held], fit["global_probability"][held])
    expected = ref_scores(counts, tr, held, fit)
    for row in scores:
        for key, value in expected[row["horizon"]].items():
            assert row[key] == pytest.approx(value, abs=1e-9)


def test_reference_geometry_matches_labels():
    rng = np.random.default_rng(134)
    base = rng.poisson(0.7, (70, 12))
    rates = rng.random((12, 30)) + 0.01
    grid = np.column_stack((np.arange(30) * 8, np.zeros(30)))
    full, half = np.arange(8), np.arange(4)
    labels, path, _ = classify(base, np.full(70, 0.005), rates, grid, full, half)
    refpath, reference = ref_geometry(base, np.full(70, 0.005), rates, grid, full)
    np.testing.assert_array_equal(path, refpath)
    for key, value in reference.items():
        assert labels["full_" + key] == pytest.approx(value)
        saved = pd.DataFrame([labels]).iloc[0]
        assert abs(float(saved["full_" + key]) - float(value)) < 1e-8
