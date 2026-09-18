from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from scripts.audit_2d_count_conditioned_prediction import folds
from scripts.forecast_tirole_content_coverage import forecast_event, relative_ids


def fitted():
    a = np.array([[0.4, 0.5, 0.1], [0.1, 0.4, 0.5], [0.5, 0.1, 0.4]])
    e = np.array([[0.4, 0.05, 0.05], [0.05, 0.4, 0.05], [0.05, 0.05, 0.4], [0.3, 0.1, 0.1], [0.1, 0.3, 0.1], [0.1, 0.1, 0.3]])
    fit = SimpleNamespace(probabilities=e, initial=np.ones(3) / 3, transition=a, occupancy=np.ones(3) / 3, global_probability=np.ones(6) / 6)
    op = NeuralOperator(fit.initial, a, fit.occupancy)
    return fit, op, OccupancyMatchedNull.from_operator(op)


def test_future_predictions_ignore_all_target_evaluation_spikes():
    fit, op, null = fitted()
    c = np.random.default_rng(2).poisson(0.8, (12, 6))
    pred, scores = forecast_event(c, np.array([0, 1, 2]), np.array([3, 4, 5]), fit, op, null)
    changed = c.copy()
    changed[:, 3:] = 10 * c[:, 3:] + 1
    again, new_scores = forecast_event(changed, np.array([0, 1, 2]), np.array([3, 4, 5]), fit, op, null)
    for h in pred:
        for name in pred[h]:
            np.testing.assert_array_equal(pred[h][name], again[h][name])
    assert scores[0]["score_dynamic"] != new_scores[0]["score_dynamic"]


def test_prediction_origin_never_uses_future_inference_spikes():
    fit, op, null = fitted()
    c = np.random.default_rng(3).poisson(0.8, (12, 6))
    pred, _ = forecast_event(c, np.array([0, 1, 2]), np.array([3, 4, 5]), fit, op, null)
    changed = c.copy()
    changed[5:, :3] = 99
    other, _ = forecast_event(changed, np.array([0, 1, 2]), np.array([3, 4, 5]), fit, op, null)
    for h in pred:
        for name in pred[h]:
            np.testing.assert_array_equal(pred[h][name][:5], other[h][name][:5])


def test_folds_hold_out_entire_events_and_one_second_guard():
    events = pd.DataFrame({"event_id": range(10), "start_s": np.arange(10) * 0.9, "end_s": np.arange(10) * 0.9 + 0.2})
    covered = []
    for _, test, cal, _ in folds(events):
        covered.extend(test.event_id.tolist())
        assert not set(test.event_id) & set(cal.event_id)
        for t in test.itertuples():
            assert ((cal.end_s + 1 <= t.start_s) | (cal.start_s >= t.end_s + 1)).all()
    assert sorted(covered) == list(range(10))


def test_unknown_or_overlapping_cells_fail():
    assert relative_ids([3, 7, 9], [3, 9]).tolist() == [0, 2]
    with pytest.raises(ValueError):
        relative_ids([3, 7, 9], [2])
    with pytest.raises(ValueError):
        relative_ids([7, 3, 9], [3])
    fit, op, null = fitted()
    with pytest.raises(ValueError, match="disjoint"):
        forecast_event(np.ones((10, 6)), [0, 1], [1, 2], fit, op, null)
