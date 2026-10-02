"""Spike-generating mechanisms, chronology and snapshot-preserving controls."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.develop_replay_pair_spike_calibration import (
    GENERATORS, copy_immigrants, generate_period, immigrant_means, make_event, run_covariates,
)
from scripts.verify_replay_pair_spike_development import reference_marginal_mean, reference_order
from scripts.diagnose_replay_pair_spike_development import baseline_support, known_copying_feature
from scripts.verify_replay_pair_spike_diagnostic import reference_architecture, reference_support

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "docs/replay_pair_spike_development_protocol.json").read_text())
PARENT = json.loads((ROOT / P["parent_protocol"]).read_text())


def test_covariates_describe_physical_motion_not_inconsistent_speed_labels():
    times, cov = run_covariates(16, .001, 100)
    estimated_speed = np.linalg.norm(np.gradient(cov["position"], .001, axis=0), axis=1)
    np.testing.assert_allclose(estimated_speed[1:-1], cov["speed"][1:-1], rtol=1e-6)
    velocity = np.gradient(cov["position"], .001, axis=0)
    angle = np.arctan2(velocity[:, 1], velocity[:, 0])
    np.testing.assert_allclose(np.exp(1j * angle[1:-1]), np.exp(1j * cov["direction"][1:-1]), atol=1e-6)
    assert np.all(np.diff(times) > 0) and np.all(cov["speed"] > 10)


def test_periods_have_distinct_clocks_but_comparable_motion():
    pre_t, pre = run_covariates(16, .001, 0)
    post_t, post = run_covariates(16, .001, 17)
    assert pre_t[-1] < post_t[0]
    for k in pre:
        np.testing.assert_array_equal(pre[k], post[k])


def test_copying_only_immigrants_does_not_create_a_recursive_cascade():
    counts = np.zeros((8, 3), int)
    counts[0, 0] = 2
    mean = counts.astype(float)
    copied, expected = copy_immigrants(counts, mean, np.arange(3), 1, 2, np.random.default_rng(1))
    assert copied[2, 1] == 2 and copied[:, 2].sum() == 0
    np.testing.assert_array_equal(copied, expected)
    np.testing.assert_array_equal(counts[:, 1:], 0)


def test_known_marginal_mean_accounts_for_expected_copied_spikes():
    counts = np.ones((8, 3), int)
    mean = np.full((8, 3), .2)
    _, expected = copy_immigrants(counts, mean, np.array([2, 0, 1]), .4, 2, np.random.default_rng(1))
    np.testing.assert_allclose(expected[2:, :2], .28)
    np.testing.assert_allclose(expected[:2], .2)
    np.testing.assert_allclose(expected[:, 2], .2)


def test_rate_drift_and_theta_shift_are_distinct_from_coupling():
    _, cov = run_covariates(1, .001, 0)
    base = immigrant_means(cov, 10, .001, "no_update", "pre", P)
    drift = immigrant_means(cov, 10, .001, "rate_only_drift", "post", P)
    np.testing.assert_allclose(drift[:, ::2], 2 * base[:, ::2])
    np.testing.assert_allclose(drift[:, 1::2], .5 * base[:, 1::2])
    shifted = immigrant_means(cov, 10, .001, "shared_theta_input", "post", P)
    assert not np.allclose(shifted[:, ::2], base[:, ::2])
    np.testing.assert_array_equal(shifted[:, 1::2], base[:, 1::2])


def test_update_changes_post_coupling_only_with_identical_immigrant_draws():
    _, cov = run_covariates(1, .001, 0)
    order = np.arange(10)
    null = generate_period(cov, order, "no_update", "pre", .001, P, np.random.default_rng(12))
    update_pre = generate_period(cov, order, "order_specific_update", "pre", .001, P, np.random.default_rng(12))
    for a, b in zip(null, update_pre, strict=True):
        np.testing.assert_array_equal(a, b)
    update_post = generate_period(cov, order, "order_specific_update", "post", .001, P, np.random.default_rng(12))
    assert update_post[0].sum() > null[0].sum()
    assert update_post[1].sum() > null[1].sum()


def test_event_shuffles_keep_participation_and_sequence_orientation():
    event, counts = make_event(np.arange(10), P, "fixed-event", np.random.default_rng(4), PARENT)
    assert counts.shape == (48, 10) and (counts.sum(axis=0) > 0).all()
    assert event["order"][0, 1] > 0
    assert event["shuffle_order"].shape == (20, 10, 10)
    np.testing.assert_array_equal(event["spike_counts"], counts.sum(axis=0))
    np.testing.assert_allclose(event["shuffle_order"], -event["shuffle_order"].swapaxes(1, 2))


def test_event_generation_uses_an_independent_random_stream():
    order = np.arange(10)
    a = make_event(order, P, "fixed", np.random.default_rng(4), PARENT)
    np.random.default_rng(99).poisson(1, size=(200, 10))
    b = make_event(order, P, "fixed", np.random.default_rng(4), PARENT)
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[0]["shuffle_order"], b[0]["shuffle_order"])


def test_protocol_contains_all_required_generators_and_no_inference_permission():
    assert tuple(P["generators"]) == GENERATORS == tuple(PARENT["calibration_required"])
    assert P["units"] >= PARENT["minimum_eligible_units"]
    assert not P["real_association_enabled"] and not P["full_calibration_complete"]


@pytest.mark.parametrize("generator", GENERATORS)
@pytest.mark.parametrize("period", ["pre", "post"])
def test_independent_known_mean_reconstruction(generator, period):
    _, covariates = run_covariates(.1, .001, 0)
    order = np.arange(10)[::-1]
    _, mean = generate_period(covariates, order, generator, period, .001, P, np.random.default_rng(4))
    bank = {"unit_ids": np.arange(10), "generating_event_order": order,
            **{f"{period}_{k}": value for k, value in covariates.items()}}
    expected = reference_marginal_mean(bank, period, generator, P, {"run_bin_s": .001})
    np.testing.assert_allclose(expected, mean, rtol=1e-14, atol=1e-16)


def test_independent_event_order_reconstruction():
    event, counts = make_event(np.arange(10), P, "fixed", np.random.default_rng(4), PARENT)
    expected = reference_order(counts, P["event_bin_s"], PARENT)
    np.testing.assert_allclose(expected, event["order"], rtol=0, atol=1e-14)


def test_supplied_architecture_is_not_claimed_to_be_measured_replay_order():
    bank = {"unit_ids": np.arange(3), "generating_event_order": np.array([2, 0, 1])}
    a, b = np.triu_indices(3, 1)
    feature = known_copying_feature(bank, a, b, "order_specific_update")
    np.testing.assert_array_equal(feature, [1, -1, 0])
    np.testing.assert_array_equal(feature, reference_architecture(bank["generating_event_order"], a, b))
    np.testing.assert_array_equal(known_copying_feature(bank, a, b, "no_update"), 0)


def test_zero_event_target_can_leave_training_baseline_row_space():
    training = np.array([[1., 1.], [-2., -2.], [3., 3.]])
    target = np.array([[1., 1.], [1., 0.]])
    support = baseline_support(training, target, np.ones(3))
    assert support["baseline_training_rank"] == 1
    assert support["baseline_max_heldout_nullspace_loading"] > .1
    rank, loading, unsupported = reference_support(training, target, np.ones(3))
    assert rank == support["baseline_training_rank"]
    assert unsupported == support["baseline_unsupported_heldout_rows"] == 1
    np.testing.assert_allclose(loading, support["baseline_max_heldout_nullspace_loading"], atol=1e-14)
    aligned = baseline_support(training, target[:1], np.ones(3))
    assert aligned["baseline_max_heldout_nullspace_loading"] < 1e-14


def test_baseline_support_is_not_based_on_test_outcomes():
    training = np.array([[1., 0.], [0., 2.], [-1., 1.]])
    support = baseline_support(training, np.array([[20., 4.]]), np.ones(3))
    assert support["baseline_training_rank"] == 2
    assert support["baseline_max_heldout_nullspace_loading"] == 0


@pytest.mark.parametrize("probability,lag", [(-.1, 2), (1.1, 2), (.5, 0), (.5, 8)])
def test_invalid_copying_cannot_become_a_valid_generator(probability, lag):
    with pytest.raises(ValueError, match="Invalid copying"):
        copy_immigrants(np.zeros((8, 3), int), np.ones((8, 3)), np.arange(3),
                        probability, lag, np.random.default_rng(1))
