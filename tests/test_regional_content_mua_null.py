import numpy as np
import pytest

from hipporeplayimm.regional_content_mua_null import (
    detector_endpoints,
    gain_profile,
    poisson_train,
    select_indices,
    window_means,
)
from scripts.measure_regional_content_mua_selection_null import draw_states


def test_endpoint_is_last_complete_four_bins():
    w, ids, crossing = detector_endpoints([
        {"event_start_s": .123, "event_end_s": .201},
        {"event_start_s": 1.94, "event_end_s": 2.011},
        {"event_start_s": 2.03, "event_end_s": 2.105},
    ])
    np.testing.assert_allclose(w, [[.178, .198], [2.085, 2.105]])
    np.testing.assert_array_equal(ids, [0, 2])
    assert crossing == 1


def test_empty_detection_is_explicit():
    w, ids, crossing = detector_endpoints([])
    assert w.shape == (0, 2) and len(ids) == crossing == 0
    means, exposure, states = window_means(w, np.array([0]), np.ones((2, 1)), np.ones(2000))
    assert means.shape == (0, 2) and len(exposure) == len(states) == 0


def test_gain_integral_and_fixed_maps():
    rates = np.array([[10., 30.], [20., 40.]])
    before = rates.copy()
    edges, gain = gain_profile(6)
    w = np.array([[.9913, 1.0113], [2.9, 2.92]])
    mean, exposure, state = window_means(w, np.array([0, 1]), rates, gain)
    overlap = np.maximum(0, np.minimum(w[:, 1, None] % 2, edges[1:]) - np.maximum(w[:, 0, None] % 2, edges[:-1]))
    np.testing.assert_allclose(exposure, overlap @ gain)
    np.testing.assert_allclose(mean, rates[:, state].T * exposure[:, None])
    np.testing.assert_array_equal(before, rates)
    _, exposure, _ = window_means(w, np.array([0, 1]), rates, np.ones(2000))
    np.testing.assert_allclose(exposure, .02)


def test_poisson_train_has_correct_rates_and_ids():
    rates = np.array([[10.], [30.]])
    _, gain = gain_profile(3)
    ids = np.array([11, 29])
    a = poisson_train(rates, ids, np.zeros(500, int), gain, np.random.default_rng(123))
    b = poisson_train(rates, ids, np.zeros(500, int), gain, np.random.default_rng(123))
    np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(a[:, 0]) >= 0) and a[:, 0].min() >= 0 and a[:, 0].max() < 1000
    count = np.array([(a[:, 1] == cell).sum() for cell in ids])
    expected = rates[:, 0]*gain.sum()*.001*500
    assert np.all(abs(count-expected) < 5*np.sqrt(expected))
    assert set(a[:, 1]) == {11, 29}


def test_independent_counts_share_mean_not_observed_counts():
    rates = np.full((5, 1), 20.)
    _, gain = gain_profile(6)
    means, _, _ = window_means([[.99, 1.01]], np.array([0]), rates, gain)
    a = np.random.default_rng(2).poisson(means, size=(5000, 5))
    b = np.random.default_rng(3).poisson(means, size=(5000, 5))
    np.testing.assert_allclose(a.mean(axis=0), means[0], atol=.08)
    np.testing.assert_allclose(b.mean(axis=0), means[0], atol=.08)
    assert not np.array_equal(a, b)


def test_hash_selection_and_balanced_calibration():
    assert np.array_equal(select_indices(np.arange(50), 20, 9), select_indices(np.arange(50), 20, 9))
    assert not np.array_equal(select_indices(np.arange(50), 20, 9), select_indices(np.arange(50), 20, 10))
    assert len(select_indices(np.arange(3), 20, 9)) == 3
    z, states = draw_states(np.array([False, False, True]), np.array([1, 0, 1]), 100, np.random.default_rng(4), True)
    assert z.sum() == 50 and set(states) == {0, 2}


def test_invalid_windows_and_gain_rejected():
    with pytest.raises(ValueError):
        window_means([[1.99, 2.01]], np.array([0, 0]), np.ones((2, 1)), np.ones(2000))
    with pytest.raises(ValueError):
        gain_profile(.5)
    with pytest.raises(ValueError):
        poisson_train(np.zeros((2, 1)), [1, 2], [0], [1.], np.random.default_rng(1))
