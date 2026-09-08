import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.assembly_predictive_control import assignment_posterior, fit_assembly, multinomial_ll, predict_heldout


def synthetic(seed=1):
    rng = np.random.default_rng(seed)
    p = np.full((12, 3), 0.01)
    for mode in range(3):
        p[mode::3, mode] = 0.25
    p /= p.sum(axis=0)
    labels = rng.integers(0, 3, size=300)
    calibration = np.array([rng.multinomial(20, p[:, mode]) for mode in labels])
    test = np.array([rng.multinomial(15, p[:, mode]) for mode in [0] * 10 + [1] * 10 + [2] * 10])
    return calibration, test


def test_em_and_recovery():
    cal, test = synthetic()
    fit = fit_assembly(cal, 3, 32)
    assert fit.converged and np.diff(fit.objective_trace).min() > -1e-6
    assert np.allclose(fit.probabilities.sum(axis=0), 1)
    scores, _ = predict_heldout(test, np.arange(len(test)) * 0.004, np.arange(6), np.arange(6, 12), fit)
    assert scores["persistent"] > scores["global"]
    assert scores["persistent"] > scores["static"]
    again = fit_assembly(cal, 3, 32)
    assert np.array_equal(fit.probabilities, again.probabilities)


def test_heldout_does_not_change_inference():
    cal, test = synthetic()
    fit = fit_assembly(cal, 3, 1)
    args = (np.arange(len(test)) * 0.004, np.arange(6), np.arange(6, 12), fit)
    scores, hashes = predict_heldout(test, *args)
    changed = test.copy()
    changed[:, 6:] = changed[::-1, 6:]
    alternate, after = predict_heldout(changed, *args)
    assert hashes == after and scores != alternate
    changed[:, 6:] = 0
    empty, after = predict_heldout(changed, *args)
    assert hashes == after
    assert max(abs(x) for x in empty.values()) < 1e-10


def test_exact_persistent_enumeration():
    ll = np.log(np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]]))
    w = np.array([0.3, 0.7])
    times = np.array([0.0, 0.004, 0.017])
    transitions = [np.exp(-dt / 0.06) * np.eye(2) + (1 - np.exp(-dt / 0.06)) * w[:, None] for dt in np.diff(times)]
    paths = list(itertools.product(range(2), repeat=3))
    scores = np.array([np.log(w[p[0]]) + sum(ll[t, p[t]] for t in range(3)) + sum(np.log(transitions[t - 1][p[t], p[t - 1]]) for t in (1, 2)) for p in paths])
    q = assignment_posterior(ll, w, times, "persistent")
    for t in range(3):
        for k in range(2):
            expected = logsumexp(scores[[p[t] == k for p in paths]]) - logsumexp(scores)
            assert q[t, k] == pytest.approx(expected)


def test_component_label_invariance():
    ll = np.array([[1.0, 3.0, 2.0], [2.0, 1.0, 4.0]])
    w, times = np.array([0.1, 0.3, 0.6]), np.array([0.0, 0.02])
    order = np.array([2, 0, 1])
    for mode in ("independent", "static", "persistent"):
        q = assignment_posterior(ll, w, times, mode)
        other = assignment_posterior(ll[:, order], w[order], times, mode)
        assert np.allclose(q[:, order], other)


@pytest.mark.parametrize("bad", [np.zeros((10, 5)), np.array([[1.0, 0.5]]), np.array([[1.0, -1.0]]), np.empty((0, 5))])
def test_invalid_calibration(bad):
    with pytest.raises(ValueError):
        fit_assembly(bad, 3, 1)


def test_multinomial_normalized():
    p = np.array([[0.25], [0.75]])
    probs = np.exp(multinomial_ll(np.array([[0, 2], [1, 1], [2, 0]]), p))
    assert probs.sum() == pytest.approx(1)


def test_failed_convergence_not_a_valid_comparator():
    cal, _ = synthetic()
    with pytest.raises(ValueError, match="converge"):
        fit_assembly(cal, 3, 1, max_iter=1)
