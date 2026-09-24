import numpy as np
import pandas as pd
import pytest

from scripts.audit_kleinman_native_adequacy import (
    bin_counts,
    complete_edges,
    crossfit_against_free_composition,
    diagnose,
    multinomial_draws,
    random_state,
    saturated_log_likelihood,
)


def templates_fixture():
    q = np.empty((3, 20, 6))
    q[0] = [0.65, 0.07, 0.07, 0.07, 0.07, 0.07]
    q[1] = [0.07, 0.65, 0.07, 0.07, 0.07, 0.07]
    for i in range(20):
        q[2, i] = q[0, i] if i < 10 else q[1, i]
    metadata = pd.DataFrame({"extent": [0, 0, 1], "start": [0, 1, 0], "end": [0, 1, 1], "direction": [0, 1, 0], "profile": ["static", "static", "linear"]})
    return np.log(q), metadata


def test_native_window_complete_bins_and_half_open_counts():
    edges = complete_edges(4.1, 4.157)
    np.testing.assert_allclose(edges, 4.1 + np.arange(6) * 0.01)
    counts = bin_counts([np.array([4.1, 4.109, 4.12, 4.151])], edges)
    assert counts.sum() == 3
    assert len(counts) == 5
    assert len(complete_edges(4.1, 4.3)) == 21
    with pytest.raises(ValueError):
        complete_edges(2, 1)


def test_replica_draws_preserve_every_population_bin_count():
    logq, _ = templates_fixture()
    totals = np.arange(20)
    draws = multinomial_draws(totals, np.exp(logq[2]), 11, np.random.default_rng(1))
    np.testing.assert_array_equal(draws.sum(axis=2), np.tile(totals, (11, 1)))
    with pytest.raises(ValueError):
        multinomial_draws(totals + 0.5, np.exp(logq[2]), 11, np.random.default_rng(1))


def test_saturated_likelihood_has_zero_bin_limit():
    c = np.array([[2, 2], [0, 0], [0, 3]])
    assert saturated_log_likelihood(c) == pytest.approx(4 * np.log(0.5))
    np.testing.assert_allclose(saturated_log_likelihood(np.stack([c, c])), [4 * np.log(0.5)] * 2)


def test_matched_template_not_automatically_rejected_and_bad_cell_pattern_is():
    logq, templates = templates_fixture()
    counts = multinomial_draws(np.full(20, 20), np.exp(logq[2]), 1, np.random.default_rng(52))[0]
    result = diagnose(counts, logq, templates, np.random.default_rng(4))
    assert result["estimated_extent"] == 1
    assert result["deviance_tail_probability"] > 0.05
    assert result["crossfit_template_minus_free"] > 0
    bad = np.zeros_like(counts)
    bad[:, 5] = 20
    rejected = diagnose(bad, logq, templates, np.random.default_rng(4))
    assert rejected["deviance_tail_probability"] == 0.01
    assert rejected["crossfit_template_minus_free"] < 0


def test_crossfit_score_can_be_rebuilt_without_test_spikes_in_training():
    logq, _ = templates_fixture()
    counts = multinomial_draws(np.arange(20) + 1, np.exp(logq[2]), 1, np.random.default_rng(92))[0]
    actual, baseline = crossfit_against_free_composition(counts, logq)
    a = b = 0.0
    for parity in [0, 1]:
        train = np.arange(20) % 2 == parity
        ll = [(counts[train] * x[train]).sum() for x in logq]
        chosen = np.flatnonzero(np.array(ll) >= max(ll) - 1e-10)
        a += np.mean([(counts[~train] * logq[i, ~train]).sum() for i in chosen])
        q = counts[train].sum(axis=0) + 0.5
        q /= q.sum()
        b += (counts[~train] * np.log(q)).sum()
    assert (actual, baseline) == pytest.approx((a, b))


def test_deterministic_disjoint_rng_namespaces():
    a = random_state("A", "s1", 4, 0).integers(0, 100000, 12)
    np.testing.assert_array_equal(a, random_state("A", "s1", 4, 0).integers(0, 100000, 12))
    assert not np.array_equal(a, random_state("A", "s1", 4, 2).integers(0, 100000, 12))
    assert not np.array_equal(a, random_state("A", "s2", 4, 0).integers(0, 100000, 12))
