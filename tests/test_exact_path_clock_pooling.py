import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_exact_path_clock_pooling import METHODS, pooled_fits


def fixture():
    metadata = pd.DataFrame(
        {
            "dataset": "test",
            "animal": "rat",
            "session": "one",
            "source_teacher": "bank",
            "scenario": 0.25,
            "row_index": range(100),
            "repeat": np.repeat([0, 1], 50),
            "event_in_population": np.tile(np.arange(50), 2),
        }
    )
    labels = np.tile(np.repeat(np.arange(5), [23, 7, 10, 5, 5]), 2)
    scores = np.log(np.where(labels[:, None] == np.arange(5), 0.999, 0.00025))
    return metadata, {method: scores.copy() for method in METHODS}


def test_pooling_keeps_same_observations_and_certifies_profiles():
    meta, matrices = fixture()
    rows = pooled_fits(meta, matrices, repeats=2, per_repeat=50)
    assert len(rows) == 3 and rows.n_events.eq(100).all()
    assert rows.fit_certified.all() and rows.interval_contains_truth.all()
    np.testing.assert_allclose(rows.phi_hat, rows.phi_hat.iloc[0])
    assert abs(rows.phi_hat.iloc[0] - 7 / 30) < 0.005
    shuffled = pooled_fits(meta.sample(frac=1, random_state=2), matrices, repeats=2, per_repeat=50)
    np.testing.assert_allclose(rows.phi_hat, shuffled.phi_hat, atol=1e-7)


def test_missing_repeat_duplicate_observations_and_nonfinite_scores_fail():
    meta, matrices = fixture()
    with pytest.raises(ValueError):
        pooled_fits(meta, matrices, repeats=3, per_repeat=50)
    wrong = meta.copy()
    wrong.loc[1, "event_in_population"] = 0
    with pytest.raises(ValueError):
        pooled_fits(wrong, matrices, repeats=2, per_repeat=50)
    wrong = meta.copy()
    wrong.loc[1, "row_index"] = 0
    with pytest.raises(ValueError):
        pooled_fits(wrong, matrices, repeats=2, per_repeat=50)
    with pytest.raises(ValueError):
        pooled_fits(meta, {"exhaustive": matrices["exhaustive"]}, repeats=2, per_repeat=50)
    matrices["exhaustive"][0, 0] = np.nan
    with pytest.raises(ValueError):
        pooled_fits(meta, matrices, repeats=2, per_repeat=50)
