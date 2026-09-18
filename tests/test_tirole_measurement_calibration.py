from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from scripts.calibrate_tirole_content_measurement import anchors, draw_counts, generate_state_counts, generate_track_counts, ordered_path
from scripts.forecast_tirole_content_coverage import forecast_event


def test_count_anchor_selection_cannot_see_outcomes():
    x = pd.DataFrame({"event_id": range(60), "epoch": ["PRE"] * 30 + ["POST"] * 30, "ripple_supported": True})
    selected = anchors(x, "test")
    assert len(selected) == 40
    x["evaluation_z"] = 999.0
    x["sequence_accepted"] = False
    assert selected.event_id.tolist() == anchors(x, "test").event_id.tolist()


def test_generators_preserve_per_group_bin_counts():
    rng = np.random.default_rng(91)
    original = rng.poisson(0.7, (12, 8))
    train = np.array([0, 2, 4, 6])
    held = np.array([1, 3, 5, 7])
    path = ordered_path(np.ones((2, 20), bool), 12, rng)
    assert (np.diff(path) >= 0).all() or (np.diff(path) <= 0).all()
    rates = rng.uniform(0.1, 10, (2, 8, 20))
    generated = generate_track_counts(original, rates, train, held, path, 1, rng)
    for ids in [train, held]:
        np.testing.assert_array_equal(generated[:, ids].sum(axis=1), original[:, ids].sum(axis=1))
    states = rng.uniform(0.1, 1, (8, 3))
    states /= states.sum(axis=0, keepdims=True)
    generated, path = generate_state_counts(original, states, train, held, np.ones(3) / 3, np.ones((3, 3)) / 3, rng)
    assert len(path) == len(original)
    for ids in [train, held]:
        np.testing.assert_array_equal(generated[:, ids].sum(axis=1), original[:, ids].sum(axis=1))


def test_invalid_totals_and_unsupported_path_fail():
    with pytest.raises(ValueError):
        draw_counts([0.3, 1], np.ones((2, 4)), np.random.default_rng(2))
    with pytest.raises(ValueError):
        ordered_path(np.zeros((2, 20), bool), 10, np.random.default_rng(2))


def test_forecast_calibration_distinguishes_temporal_generator_from_null():
    transition = np.array([[0.1, 0.85, 0.05], [0.05, 0.1, 0.85], [0.85, 0.05, 0.1]])
    emissions = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 2) / 2
    fit = SimpleNamespace(initial=np.ones(3) / 3, transition=transition, occupancy=np.ones(3) / 3, probabilities=emissions, global_probability=np.ones(6) / 6)
    op = NeuralOperator(fit.initial, transition, fit.occupancy)
    null = OccupancyMatchedNull.from_operator(op)
    results = {}
    for name, matrix in [("dynamic", transition), ("null", null.step(np.eye(3)))]:
        gains = []
        for rep in range(100):
            c, _ = generate_state_counts(np.ones((20, 6), int), emissions, [0, 1, 2], [3, 4, 5], fit.initial, matrix, np.random.default_rng(rep))
            _, scores = forecast_event(c, np.arange(3), np.arange(3, 6), fit, op, null)
            primary = scores[1]
            gains.append(primary["score_dynamic"] - primary["score_matched_own"])
        results[name] = np.mean(gains)
    assert results["dynamic"] > 0
    assert results["null"] < 0
