from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.audit_tirole_local_rest_baseline import baseline_counts, batch_odds, local_null_counts, rest_masks
from scripts.verify_tirole_content_bank import direct_posterior


def session_fixture():
    t = np.arange(0.0, 300.0, 0.04)
    pos = np.full((2, len(t)), np.nan)
    pos[0, (t >= 100) & (t < 150)] = 40
    pos[1, (t >= 160) & (t <= 200)] = 60
    st = np.array([20.0, 218.0, 220.0, 222.0, 240.0])
    return SimpleNamespace(
        times=t,
        positions=pos,
        sleepbox=np.ones(len(t), bool),
        speed=np.zeros(len(t)),
        spike_samples=np.searchsorted(t, st),
        spike_units=np.array([0, 0, 1, 1, 0]),
        unit_ids=np.array([0, 1]),
    )


def test_baseline_excludes_event_and_guards_and_other_epoch():
    data = session_fixture()
    e = pd.DataFrame({"start_s": [220.0], "end_s": [220.2]})
    masks = rest_masks(data, e)
    assert not masks["POST"][data.times <= 200].any()
    assert not masks["PRE"][data.times >= 100].any()
    assert not masks["POST"][(data.times >= 219.0) & (data.times <= 221.2)].any()
    _, c = baseline_counts(data, masks["POST"], 0, len(data.times))
    np.testing.assert_equal(c, [2, 1])
    # Arbitrarily many target spikes do not enter the outside-event rate estimate.
    data.spike_samples = np.r_[data.spike_samples, np.repeat(np.searchsorted(data.times, 220.0), 200)]
    data.spike_units = np.r_[data.spike_units, np.ones(200, int)]
    _, changed = baseline_counts(data, masks["POST"], 0, len(data.times))
    np.testing.assert_equal(c, changed)


def test_null_preserves_population_count_envelope():
    c = np.array([[0, 0], [4, 1], [2, 0]])
    x = local_null_counts(c, np.array([20, 1]), np.random.default_rng(3))
    assert x.shape == (200, 3, 2)
    np.testing.assert_equal(x[0], c)
    np.testing.assert_equal(x.sum(axis=2), np.tile(c.sum(axis=1), (200, 1)))


def test_batch_readout_matches_independent_likelihood():
    rng = np.random.default_rng(4)
    c = rng.poisson(2.0, (7, 6, 4))
    c[:, 0] = 0
    rates = rng.uniform(0.2, 6.0, (2, 4, 8))
    valid = np.ones((2, 8), bool)
    valid[0, 0] = False
    for cond in [False, True]:
        actual = batch_odds(c, rates, valid, cond)
        expected = []
        for x in c:
            p = direct_posterior(x[x.sum(axis=1) > 0], rates, valid, cond)
            mass = p.sum(axis=(0, 2))
            expected.append(np.log(mass[0] / mass[1]))
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.isnan(batch_odds(np.zeros_like(c), rates, valid)).all()


def test_invalid_null_counts_rejected():
    with pytest.raises(ValueError):
        local_null_counts(np.array([[0.5, 1]]), np.array([2.0, 3.0]), np.random.default_rng(1))
