import numpy as np
import pytest

from hipporeplayimm.two_track_content import content_readout
from scripts.audit_tirole_evaluation_identity import context_odds, matched_permutations, null_summary


def test_matching_is_deterministic_and_preserves_RUN_rate_group():
    rates = np.array([0.7, 8, 0.8, 0.9, 9, 1, 10, 1.1, 11, 1.2, 12, 13])
    p, g = matched_permutations(rates, seed=9)
    again, _ = matched_permutations(rates, seed=9)
    assert np.array_equal(p, again)
    assert np.all(g[p] == g)
    assert np.all(np.sort(p, axis=1) == np.arange(len(rates)))
    assert len(np.unique(p, axis=0)) > 100


@pytest.mark.parametrize("conditional", [False, True])
def test_observed_and_each_permutation_match_original_readout(conditional):
    rng = np.random.default_rng(123)
    rates = rng.uniform(0.2, 6, (2, 6, 8))
    counts = rng.poisson(0.7, (12, 6))
    valid = np.ones((2, 8), bool)
    valid[:, -1] = False
    permutations, _ = matched_permutations(np.ones(6), seed=3)
    odds = context_odds(counts, rates, valid, permutations, conditional)
    swaps = rng.integers(0, 2, (39, 6)).astype(bool)
    for i, ids in enumerate(np.vstack([np.arange(6), permutations[:5]])):
        expected = content_readout(counts, rates[:, ids], valid, swaps, conditional)["log_odds"]
        assert odds[i] == pytest.approx(expected, abs=1e-12)


def test_correct_identity_carries_track_signal_beyond_rate_null():
    rates = np.ones((2, 12, 10))
    rates[0, :6], rates[1, 6:] = 10, 10
    counts = np.tile(np.r_[np.ones(6), np.zeros(6)], (8, 1))
    order = np.ravel(np.c_[np.arange(6), np.arange(6, 12)])
    rates, counts = rates[:, order], counts[:, order]
    p, _ = matched_permutations(np.ones(12), seed=123)
    for conditional in (False, True):
        result = null_summary(context_odds(counts, rates, np.ones((2, 10), bool), p, conditional))
        assert result["identity_excess_log_odds"] > 0
        assert result["identity_z_log_odds"] > 1


def test_empty_counts_are_missing_not_pass():
    p, _ = matched_permutations(np.ones(6))
    odds = context_odds(np.zeros((5, 6)), np.ones((2, 6, 4)), np.ones((2, 4), bool), p)
    assert np.isnan(odds).all()
    assert np.isnan(null_summary(odds)["identity_z_log_odds"])


def test_identical_context_maps_cannot_support_track_content():
    p, _ = matched_permutations(np.ones(6))
    odds = context_odds(np.ones((5, 6)), np.ones((2, 6, 4)), np.ones((2, 4), bool), p)
    assert np.all(odds == 0)
    assert np.isnan(null_summary(odds)["identity_z_log_odds"])


def test_invalid_or_duplicated_null_indices_fail():
    with pytest.raises(ValueError, match="permute"):
        context_odds(np.ones((5, 6)), np.ones((2, 6, 4)), np.ones((2, 4), bool), np.zeros((39, 6), int))
    with pytest.raises(ValueError, match="positive"):
        matched_permutations(np.zeros(6))
