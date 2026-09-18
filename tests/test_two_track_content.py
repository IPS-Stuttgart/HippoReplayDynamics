from dataclasses import replace

import numpy as np
import pytest
from test_tirole_two_track import synthetic_session

from hipporeplayimm.tirole_two_track import decode_counts
from hipporeplayimm.two_track_content import classify_sequence, content_readout, event_bin_counts, field_shift_posteriors, nested_subsets, split_populations, weighted_correlation


def maps_and_counts():
    rng = np.random.default_rng(72)
    x = np.arange(20)
    peaks = np.arange(30) * 19 / 29
    maps = 0.1 + 20 * np.exp(-((x[None, :] - peaks[:, None]) ** 2) / (2 * 1.3**2))
    rates = np.stack([maps, maps[rng.permutation(30)]])
    counts = rng.poisson(rates[0].T * 0.3)
    valid = np.ones((2, 20), bool)
    shifts = rng.integers(0, 20, (199, 2, 30))
    permutations = np.array([rng.permutation(20) for _ in range(199)])
    return rates, counts, valid, shifts, permutations


def test_populations_disjoint_and_nested():
    ids = np.arange(40) * 5
    detector, splits = split_populations(ids, "session")
    d2, s2 = split_populations(ids, "session")
    np.testing.assert_equal(detector, d2)
    for i, (a, b) in enumerate(splits):
        assert not set(a) & set(b) and not set(a) & set(detector) and not set(b) & set(detector)
        assert set(a) | set(b) | set(detector) == set(ids)
        np.testing.assert_equal(a, s2[i][0])
        subsets = nested_subsets(a, "session", i, 0)
        assert set(subsets[0.25]) <= set(subsets[0.5]) <= set(subsets[0.75]) <= set(subsets[1.0])


def test_weighted_correlation_known_linear_and_stationary():
    p = np.stack([np.eye(20), np.tile(np.eye(20)[10], (20, 1))], axis=1)
    np.testing.assert_allclose(weighted_correlation(p, np.arange(20)), [1, 0], atol=1e-12)
    np.testing.assert_allclose(weighted_correlation(np.stack([p, p * 2]), np.arange(20)), [[1, 0], [1, 0]], atol=1e-12)


def test_field_shift_zero_equals_original_decoder():
    rates, counts, valid, _, _ = maps_and_counts()
    null = field_shift_posteriors(counts, rates, valid, np.zeros((2, 2, 30), int))
    np.testing.assert_allclose(null[0], decode_counts(counts, rates, 0.02, valid))


def test_established_shuffle_criterion_accepts_ordered_fixture():
    rates, counts, valid, shifts, perm = maps_and_counts()
    r = classify_sequence(counts, rates, valid, np.arange(20), shifts, perm)
    assert r["sequence_accepted"] and r["inferred_track"] == 1
    assert r["track1_p_time"] < 0.025 and r["track1_p_field"] < 0.025


def test_time_scramble_rejected_for_fixed_fixture():
    rates, counts, valid, shifts, perm = maps_and_counts()
    shuffled = counts[np.random.default_rng(93).permutation(20)]
    r = classify_sequence(shuffled, rates, valid, np.arange(20), shifts, perm)
    assert not r["sequence_accepted"]


def test_zero_support_fails_nonvacuously():
    rates, counts, valid, shifts, perm = maps_and_counts()
    r = classify_sequence(counts * 0, rates, valid, np.arange(20), shifts, perm)
    assert not r["sequence_eligible"] and not r["sequence_accepted"]


def test_bad_time_null_rejected():
    rates, counts, valid, shifts, perm = maps_and_counts()
    perm[0, :] = 0
    with pytest.raises(ValueError):
        classify_sequence(counts, rates, valid, np.arange(20), shifts, perm)


def test_content_is_sequence_independent_and_track_sensitive():
    rates, counts, valid, _, _ = maps_and_counts()
    swaps = np.random.default_rng(28).integers(0, 2, (199, 30)).astype(bool)
    r = content_readout(counts, rates, valid, swaps)
    rev = content_readout(counts[::-1], rates, valid, swaps)
    np.testing.assert_allclose([r["log_odds"], r["z_log_odds"]], [rev["log_odds"], rev["z_log_odds"]])
    assert r["z_log_odds"] > 2 and r["track2_probability"] < 0.5
    flipped = content_readout(counts, rates[::-1], valid, swaps)
    assert flipped["log_odds"] < 0


def test_conditional_count_removes_global_rate_difference():
    rng = np.random.default_rng(14)
    template = rng.uniform(1, 4, (1, 15, 20))
    rates = np.concatenate([template, template * 3])
    counts = rng.poisson(1, (10, 15))
    swaps = rng.integers(0, 2, (99, 15)).astype(bool)
    r = content_readout(counts, rates, np.ones((2, 20), bool), swaps, True)
    assert abs(r["log_odds"]) < 1e-10


def test_empty_evaluation_is_missing_not_zero_evidence():
    rates, counts, valid, _, _ = maps_and_counts()
    r = content_readout(counts * 0, rates, valid, np.zeros((99, 30), bool))
    assert not r["content_supported"] and np.isnan(r["log_odds"])


def test_counts_no_short_last_bin_and_end_exclusive():
    s = synthetic_session()
    st = np.array([1.0, 1.01, 1.02, 1.04, 1.059, 1.061])
    s = replace(s, spike_times=st, spike_units=np.zeros(len(st), int))
    counts = event_bin_counts(s, 1.0, 1.065)
    assert counts.shape == (3, 30)
    assert counts.sum() == 5


def test_counts_exact_boundaries_at_large_recording_timestamps():
    s = synthetic_session()
    start = 436982.738098
    edges = start + np.arange(5) * 0.02
    s = replace(s, spike_times=edges, spike_units=np.zeros(5, int))
    counts = event_bin_counts(s, start, start + 0.085)
    np.testing.assert_equal(counts[:, 0], [1, 1, 1, 1])
