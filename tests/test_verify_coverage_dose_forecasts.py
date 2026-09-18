"""Independent verifier must agree without calling production aggregation."""

import numpy as np
from test_coverage_dose_forecast import rows

from hipporeplayimm.coverage_dose_forecast import event_tables, summaries
from hipporeplayimm.independent_rejected_forecast import forecast_distributions, score_predictions
from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from scripts.verify_coverage_dose_forecasts import compare_frames, independent_events, independent_summary, reference_score
from scripts.verify_independent_rejected_forecasts import ref_null


def test_independent_aggregation_and_paired_changes():
    expected = event_tables(rows())
    actual = independent_events(rows())
    keys = ["dataset", "animal", "session", "event_id", "arm", "fraction", "group"]
    compare_frames(expected, actual, keys, [c for c in actual if c not in keys])
    a, s = independent_summary(actual)
    prod = summaries(expected)
    keys = ["dataset", "animal", "arm", "fraction", "group"]
    compare_frames(prod[1], a, keys, [c for c in a if c not in keys])
    keys = ["dataset", "arm", "fraction", "group", "metric"]
    compare_frames(prod[2], s, keys, [c for c in s if c not in keys])


def test_independent_future_prediction_uses_own_filter():
    rng = np.random.default_rng(427)
    a = rng.random((5, 5)) + 0.05
    a /= a.sum(axis=1, keepdims=True)
    p = rng.random((12, 5)) + 0.02
    fit = {"initial": np.full(5, 0.2), "transition": a, "occupancy": np.full(5, 0.2), "probabilities": p, "global_probability": np.ones(12)}
    counts = rng.poisson(0.4, (15, 12))
    train, held = np.arange(0, 8, 2), np.arange(8, 12)
    op = NeuralOperator(fit["initial"], a, fit["occupancy"])
    null = OccupancyMatchedNull.from_operator(op)
    actual = score_predictions(forecast_distributions(counts[:, train], p[train], op, null, (2,)), counts[:, held], p[held], np.ones(4))[0]
    expected = reference_score(counts, train, held, fit, ref_null(a))
    for key in expected:
        np.testing.assert_allclose(expected[key], actual[key], atol=1e-9)
