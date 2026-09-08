import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.multicontext_prediction import predict_contexts
from scripts.verify_tanni_multicontext_prediction import CONTRASTS, csv_ids, guarded_ids, primary_pass, recount, reference_prediction


def test_independent_equations_reproduce_repeated_context_prediction():
    rng = np.random.default_rng(7)
    maps = [rng.uniform(0.01, 10, size=(8, n)) for n in (2, 8, 12, 18, 4)]
    compositions = [m.mean(axis=1) for m in maps]
    counts = rng.poisson(1, (6, 8))
    kwargs = (counts, maps, compositions, np.arange(5), np.arange(5, 8), list("ABCDA"), "A", np.arange(1, 9))
    produced = predict_contexts(*kwargs)
    direct = reference_prediction(*kwargs)
    for k, value in direct.items():
        assert produced[k] == pytest.approx(value, abs=1e-10)


def test_independent_predictions_ignore_heldout_for_context():
    rng = np.random.default_rng(13)
    maps = [rng.uniform(0.01, 5, (4, 6)) for _ in range(2)]
    compositions = [m.mean(axis=1) for m in maps]
    counts = np.array([[2, 4, 1, 3], [3, 1, 4, 0]])
    a = reference_prediction(counts, maps, compositions, [0, 1], [2, 3], ["A", "B"], "A", np.ones(4))
    counts[:, 2:] = 10
    b = reference_prediction(counts, maps, compositions, [0, 1], [2, 3], ["A", "B"], "A", np.ones(4))
    assert a["iid_p_A"] == b["iid_p_A"]
    assert a["mix_iid"] != b["mix_iid"]


def test_recount_keeps_half_open_clock_and_silent_cells():
    counts = recount([np.array([0, 0.02, 0.039, 0.04]), np.array([])], np.array([0, 0.02, 0.04]))
    np.testing.assert_array_equal(counts, [[1, 0], [2, 0]])


def test_guard_excludes_target_fold_and_temporal_neighbors():
    events = pd.DataFrame({"event_id": np.arange(5), "start_s": [0, 0.4, 3, 6, 9], "end_s": [0.2, 0.6, 3.2, 6.2, 9.2]})
    test, calibration, excluded = guarded_ids(events, 0)
    assert test == {0} and calibration == {2, 3, 4} and excluded == {1}
    assert csv_ids(np.nan) == set()
    assert csv_ids("1.0,2") == {1, 2}


def test_nonspatial_failure_cannot_be_rescued_by_context_improvement():
    summary = pd.DataFrame({"metric": list(CONTRASTS)[:3], "mean": [1, 2, 3], "ci_low": [0.2, 0.3, 0.4], "positive_animals": [5, 5, 5]}).set_index("metric")
    assert primary_pass(summary)
    summary.loc["mix_iid_minus_event_global", "ci_low"] = -1
    assert not primary_pass(summary)
    summary.loc["mix_iid_minus_event_global", "ci_low"] = 0.4
    summary.loc["mix_iid_minus_current_iid", "positive_animals"] = 4
    assert not primary_pass(summary)
