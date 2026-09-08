import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.conditional_spatial_prediction import SpatialPredictionContext, identity_likelihood
from scripts.audit_2d_predictive_order_map import MAPS, contrasts, decision, permuted_times, shuffled_indices, shuffled_prediction
from scripts.verify_2d_count_conditioned_prediction import reference_posterior


def test_permutation_preserves_counts_and_partial_exposure():
    edges = np.array([2.0, 2.02, 2.04, 2.047])
    permutation = np.array([2, 0, 1])
    clock = permuted_times(edges, permutation)
    np.testing.assert_allclose(clock, [0.0035, 0.017, 0.037])
    counts = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(counts[permutation].sum(axis=0), counts.sum(axis=0))
    assert set(map(tuple, counts)) == set(map(tuple, counts[permutation]))
    assert np.isclose(np.diff(edges)[permutation].sum(), edges[-1] - edges[0])


def test_deterministic_order_includes_identity_and_duplicates():
    orders = [shuffled_indices(("d", "a", "s"), 1, 2, k) for k in range(20)]
    assert any(np.array_equal(p, [0, 1]) for p in orders)
    assert len({tuple(p) for p in orders}) == 2
    for k, p in enumerate(orders):
        np.testing.assert_array_equal(p, shuffled_indices(("d", "a", "s"), 1, 2, k))


def test_partial_bin_prediction_independently_reconstructed_and_nonleaking():
    rng = np.random.default_rng(29)
    x = np.array(list(itertools.product([0.0, 8.0, 16.0], repeat=2)))
    context = SpatialPredictionContext(x)
    rates = rng.uniform(0.1, 3, (8, 9))
    counts = rng.poisson(0.8, (7, 8))
    tll, hll = identity_likelihood(counts[:, :5], rates[:5]), identity_likelihood(counts[:, 5:], rates[5:])
    p = np.array([6, 2, 5, 3, 4, 0, 1])
    times = permuted_times(np.r_[np.arange(7) * 0.02, 0.133], p)
    map_p = rng.permutation(9)
    scores = shuffled_prediction(tll[p], hll[p], times, context, map_p)
    changed = shuffled_prediction(tll[p], np.zeros_like(hll[p]), times, context, map_p)
    for row, other, order in zip(scores, changed, (np.arange(9), map_p), strict=True):
        assert row["training_imm_posterior_sha256"] == other["training_imm_posterior_sha256"]
        for model in ("first_order_imm", "diffusion"):
            post = reference_posterior(tll[p][:, order], x, times, model == "first_order_imm")
            np.testing.assert_allclose(row[f"score_{model}"], logsumexp(post + hll[p][:, order], axis=1).sum(), atol=1e-9)
            assert abs(other[f"score_{model}"]) < 1e-10


def fixture_scores():
    original, shuffled = [], []
    for split in range(5):
        for name in MAPS:
            base = {
                "dataset": "test",
                "animal": "a",
                "session": "s",
                "event_id": 1,
                "split": split,
                "map": name,
                "n_heldout_spikes": 10,
                "posterior_unchanged": True,
                "heldout_used_for_inference": False,
                "score_iid_position": -20.0,
                "score_static_location": -21.0,
            }
            original.append(base | {"score_first_order_imm": -10.0 if name == "real" else -12.0, "score_diffusion": -11.0})
            for k in range(3):
                shuffled.append(base | {"shuffle": k, "score_first_order_imm": -16.0, "score_diffusion": -15.0})
    return pd.DataFrame(original), pd.DataFrame(shuffled)


def test_paired_factorial_before_event_aggregation():
    original, shuffled = fixture_scores()
    splits, events = contrasts(shuffled, original, k=3)
    values = events.set_index("contrast")
    assert len(splits) == 90 and len(events) == 18
    assert values.loc["first_order_imm__real_order_advantage", "delta"] == 6
    assert values.loc["first_order_imm__wrong_order_advantage", "delta"] == 4
    assert values.loc["first_order_imm__order_map_interaction", "delta"] == 2
    assert values.loc["first_order_imm__order_map_interaction", "delta_per_heldout_spike"] == 0.2


@pytest.mark.parametrize("damage", ["empty", "missing_shuffle", "missing_map", "duplicate", "leak"])
def test_invalid_factorial_fails(damage):
    original, shuffled = fixture_scores()
    if damage == "empty":
        shuffled = shuffled.iloc[:0]
    elif damage == "missing_shuffle":
        shuffled = shuffled.iloc[1:]
    elif damage == "missing_map":
        shuffled = shuffled[shuffled["map"].eq("real")]
    elif damage == "duplicate":
        shuffled = pd.concat([shuffled, shuffled.iloc[:1]])
    else:
        shuffled.loc[0, "heldout_used_for_inference"] = True
    with pytest.raises(ValueError):
        contrasts(shuffled, original, k=3)


def test_empty_or_negative_primary_is_not_a_pass():
    rows = []
    for dataset, rats in (("pfeiffer_foster", 4), ("tanni2022", 5)):
        for name in ("real_order_advantage", "order_map_interaction"):
            rows.append({"dataset": dataset, "contrast": "first_order_imm__" + name, "animals": rats, "positive_animals": rats, "mean": 1.0, "ci_low": 0.5})
    frame = pd.DataFrame(rows)
    assert decision(frame).order_and_adjacency_gate_passed.all()
    frame.loc[3, "ci_low"] = -0.1
    assert not decision(frame).iloc[1].order_and_adjacency_gate_passed
    assert not decision(frame).parent_replication_gate_changed.any()
    with pytest.raises(ValueError):
        decision(frame.iloc[:0])


def test_known_ordered_generator_has_predictive_order_advantage():
    rng = np.random.default_rng(5912)
    x = np.column_stack([np.arange(8) * 8.0, np.zeros(8)])
    rates = 0.005 + np.exp(-((x[:, None] - x[None, :]) ** 2).sum(axis=2) / (2 * 12**2))
    probability = rates / rates.sum(axis=0)
    path = np.r_[np.arange(8), np.arange(6, -1, -1)]
    times = np.arange(len(path)) * 0.02
    context = SpatialPredictionContext(x)
    differences = []
    for _ in range(60):
        train = np.array([rng.multinomial(2, probability[:, p]) for p in path])
        held = np.array([rng.multinomial(3, probability[:, p]) for p in path])
        tll, hll = identity_likelihood(train, rates), identity_likelihood(held, rates)
        permutation = rng.permutation(len(path))
        original = context.infer(tll, times)[0]
        shuffled = context.infer(tll[permutation], times)[0]
        score = lambda q, target: logsumexp(q + target, axis=1).sum()
        differences.append(score(original["first_order_imm"], hll) - score(shuffled["first_order_imm"], hll[permutation]))
        np.testing.assert_allclose(score(original["iid_position"], hll), score(shuffled["iid_position"], hll[permutation]), atol=1e-10)
    assert np.mean(differences) > 0


def test_identical_population_snapshots_have_no_order_effect():
    x = np.array([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0]])
    train = np.tile(np.array([-1.0, -2.0, -4.0]), (5, 1))
    held = np.tile(np.array([-2.0, -1.0, -3.0]), (5, 1))
    context = SpatialPredictionContext(x)
    p = np.array([3, 0, 2, 1, 4])
    a = shuffled_prediction(train, held, np.arange(5) * 0.02, context, np.arange(3))
    b = shuffled_prediction(train[p], held[p], np.arange(5) * 0.02, context, np.arange(3))
    assert a == b
