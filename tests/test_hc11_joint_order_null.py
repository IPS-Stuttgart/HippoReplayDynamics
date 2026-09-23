import itertools

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import logsumexp

from scripts.calibrate_hc11_joint_order_null import (
    constrained_mstep,
    estep,
    fit_model,
    relabelling_check,
    routing_invariance_error,
    routing_objective,
)
from scripts.simulate_hc11_phase_order_identifiability import equilibrium, generate, make_world


def test_joint_mstep_gradient():
    rng = np.random.default_rng(3)
    counts = rng.uniform(.1, 10, (2, 4, 4))
    x = rng.normal(0, .3, 20)
    error = check_grad(lambda v: routing_objective(v, counts)[0],
                       lambda v: routing_objective(v, counts)[1], x)
    assert error < 1e-6


def test_shared_order_can_represent_changed_occupancy_and_dwell():
    pre, post = make_world(8, "occupancy_dwell_only")
    expected = np.array([equilibrium(w[0])[:, None] * w[0] * 1e5 for w in (pre, post)])
    fitted, _, gradient = constrained_mstep(expected)
    np.testing.assert_allclose(fitted, np.array([pre[0], post[0]]), atol=2e-5, rtol=0)
    assert routing_invariance_error(fitted) < 1e-10
    assert gradient < 1e-6


def test_reversed_ring_violates_shared_order_constraint():
    pre, post = make_world(8, "order_changed")
    assert routing_invariance_error(np.array([pre[0], post[0]])) > 1


def test_expected_counts_against_full_path_enumeration():
    a = np.array([[.6, .3, .1], [.2, .4, .4], [.3, .1, .6]])
    pi = np.array([.2, .5, .3])
    ll = np.log(np.array([[.2, .5, .3], [.8, .1, .1], [.1, .2, .7]]))
    paths = list(itertools.product(range(3), repeat=3))
    weights = [pi[p[0]] * np.exp(sum(ll[t, p[t]] for t in range(3))) * a[p[0], p[1]] * a[p[1], p[2]] for p in paths]
    expected_starts, expected_transitions = np.zeros(3), np.zeros((3, 3))
    for p, w in zip(paths, weights, strict=True):
        expected_starts[p[0]] += w / sum(weights)
        for t in (0, 1):
            expected_transitions[p[t], p[t + 1]] += w / sum(weights)
    logp, starts, transitions = estep([ll], pi, a)
    assert logp == pytest.approx(np.log(sum(weights)), abs=1e-12)
    np.testing.assert_allclose(starts, expected_starts, atol=1e-12)
    np.testing.assert_allclose(transitions, expected_transitions, atol=1e-12)


def test_no_cross_event_transition_in_estep():
    a = np.full((3, 3), 1 / 3)
    ll = np.log(np.array([[.2, .5, .3]]))
    _, starts, transitions = estep([ll, ll], np.full(3, 1 / 3), a)
    assert starts.sum() == pytest.approx(2)
    assert transitions.sum() == 0


def test_free_emission_maps_hide_true_reversal():
    result = relabelling_check()
    assert result["observationally_equivalent"]
    assert result["max_absolute_log_score_difference"] < 1e-10


def test_joint_em_monotonic_and_nested():
    world = make_world(19, "order_changed")
    rng = np.random.default_rng(19)
    data = [generate(rng, w, 20, 20) for w in world]
    emissions = [w[1] for w in world]
    common = fit_model(data, emissions, common_order=True)
    separate = fit_model(data, emissions, common_order=False, start=common)
    assert common.converged and separate.converged
    assert np.min(np.diff(common.history)) >= -1e-7
    assert np.min(np.diff(separate.history)) >= -1e-7
    assert separate.history[-1] > common.history[-1]
    assert routing_invariance_error(common.transition) < 1e-10


@pytest.mark.parametrize("counts", [np.zeros((2, 3, 3)), np.ones((1, 3, 3)), np.full((2, 3, 3), np.nan)])
def test_invalid_mstep_counts(counts):
    with pytest.raises(ValueError):
        constrained_mstep(counts)


def test_posterior_normalization_in_enumeration():
    ll = np.array([[-1000., -1002., -999.]])
    _, starts, _ = estep([ll], np.full(3, 1 / 3), np.full((3, 3), 1 / 3))
    np.testing.assert_allclose(starts, np.exp(ll[0] - logsumexp(ll[0])), atol=1e-12)
