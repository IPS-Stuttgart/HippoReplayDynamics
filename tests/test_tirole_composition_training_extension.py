import numpy as np
import pandas as pd

from scripts.expand_tirole_composition_training import fit_training, select_training
from scripts.expand_tirole_composition_training import test_likelihoods as lookup_likelihoods


def test_training_selection_excludes_test_windows_and_guards():
    events = pd.DataFrame(
        {"event_id": range(7), "start_s": [0.0, 1.0, 3.0, 5.0, 6.0, 8.0, 12.0], "end_s": [0.2, 1.2, 3.2, 5.2, 6.2, 8.2, 12.2], "ripple_supported": True, "epoch": "POST"}
    )
    selected = select_training(events, events.iloc[[1, 4]], "test")
    assert set(selected.event_id) == {2, 5, 6}


def test_selection_score_blind_deterministic_under_row_order():
    events = pd.DataFrame({"event_id": range(200), "start_s": np.arange(200) * 3.0, "end_s": np.arange(200) * 3.0 + 0.2, "ripple_supported": True, "epoch": "POST"})
    test = events.iloc[:10]
    a = select_training(events, test, "test")
    events["model_score"] = np.random.default_rng(4).normal(size=len(events))
    events["sequence_accepted"] = True
    b = select_training(events.sample(frac=1, random_state=2), test, "test")
    assert len(a) == 80
    assert a.event_id.tolist() == b.event_id.tolist()


def test_training_histogram_uses_training_not_test_values():
    train = np.zeros((20, 5, 2))
    train[:, :, 1] = 1.0
    before = fit_training(train, np.ones((1, 20)))
    q = np.full((10, 5, 2), 0.5)
    a = lookup_likelihoods(q, before)
    q[:] = 1.0
    b = lookup_likelihoods(q, before)
    np.testing.assert_array_equal(before, fit_training(train, np.ones((1, 20))))
    assert not np.array_equal(a, b)


def test_training_support_counts_unique_anchors_not_bootstrap_multiplicity():
    q = np.zeros((20, 5, 2))
    q[:, :, 1] = 1.0
    weights = np.zeros((1, 20))
    weights[0, :4] = 5
    assert np.isnan(fit_training(q, weights)).all()
    weights[0, 4] = 1
    assert np.isfinite(fit_training(q, weights)).all()
