import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.conditional_spatial_prediction import SpatialPredictionContext, identity_likelihood, score_event
from scripts.audit_2d_count_conditioned_prediction import contrasts, folds, pool_counts, technical_gates


def dense_kernel(x, sigma):
    d2 = ((x[:, None] - x[None, :]) ** 2).sum(axis=2)
    weights = np.exp(-d2 / (2 * sigma**2)) * (d2 <= (3 * sigma) ** 2)
    return weights / weights.sum(axis=0, keepdims=True)


def test_exact_imm_enumeration_variable_bins():
    x = np.array([[0, 0], [8, 0], [0, 8.0]])
    ll = np.log(np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5], [0.1, 0.7, 0.2]]))
    times = np.array([0.01, 0.03, 0.045])
    matrices = []
    for dt in np.diff(times):
        kernels = [dense_kernel(x, 2), dense_kernel(x, 60 * np.sqrt(dt)), np.full((3, 3), 1 / 3)]
        diagonal = np.exp(-dt / 0.06)
        modes = np.full((3, 3), (1 - diagonal) / 2)
        np.fill_diagonal(modes, diagonal)
        matrices.append(np.block([[modes[src, dst] * kernels[dst] for src in range(3)] for dst in range(3)]))
    expected = np.zeros((3, 9))
    for sequence in itertools.product(range(9), repeat=3):
        p = np.exp(sum(ll[t, s % 3] for t, s in enumerate(sequence))) / 9
        p *= matrices[0][sequence[1], sequence[0]] * matrices[1][sequence[2], sequence[1]]
        for t, s in enumerate(sequence):
            expected[t, s] += p
    expected /= expected.sum(axis=1, keepdims=True)
    post, mass = SpatialPredictionContext(x).infer(ll, times)
    np.testing.assert_allclose(np.exp(post["first_order_imm"]), expected.reshape(3, 3, 3).sum(axis=1), atol=1e-10)
    np.testing.assert_allclose(mass, expected.reshape(3, 3, 3).sum(axis=2), atol=1e-10)


def test_cell_identity_distribution_normalized():
    k = np.array([[0, 3], [1, 2], [2, 1], [3, 0]])
    r = np.array([[1.0, 4.0], [3.0, 1.0]])
    np.testing.assert_allclose(np.exp(identity_likelihood(k, r)).sum(axis=0), 1)


def test_no_heldout_leakage_and_map_independent_invariance():
    rng = np.random.default_rng(12)
    counts = rng.poisson(1, (8, 6))
    rates = rng.uniform(0.1, 5, (6, 4))
    ctx = SpatialPredictionContext(np.array([[0, 0], [8, 0], [0, 8], [8, 8]]))
    args = (np.arange(8) * 0.02, rates, [0, 1, 2, 3], [4, 5], ctx, np.array([2, 0, 3, 1]), rates.mean(axis=1))
    first = score_event(counts, *args)
    counts[:, 4:] = 0
    second = score_event(counts, *args)
    for a, b in zip(first, second, strict=True):
        assert a["training_imm_posterior_sha256"] == b["training_imm_posterior_sha256"]
        assert abs(b["score_first_order_imm"]) < 1e-10
    assert np.isclose(first[0]["score_iid_position"], first[1]["score_iid_position"])
    assert np.isclose(first[0]["score_static_location"], first[1]["score_static_location"])


def test_pooling_preserves_partial_and_counts():
    c = np.arange(21).reshape(7, 3)
    counts, times, edges = pool_counts(c, np.arange(7) * 0.005, np.array([0.005] * 6 + [0.002]))
    np.testing.assert_array_equal(counts, [c[:4].sum(axis=0), c[4:].sum(axis=0)])
    np.testing.assert_allclose(edges, [0, 0.02, 0.032])
    np.testing.assert_allclose(times, [0.01, 0.026])


def test_cross_event_folds_keep_test_and_guard_calibration():
    events = pd.DataFrame({"event_id": range(10), "start_s": np.arange(10) * 0.9, "end_s": np.arange(10) * 0.9 + 0.1})
    result = folds(events)
    assert sorted(i for _, test, _, _ in result for i in test.event_id) == list(range(10))
    for _, test, cal, excluded in result:
        assert not set(test.event_id) & set(cal.event_id)
        assert len(test) + len(cal) + len(excluded) == 10
        for t in test.itertuples():
            assert ((cal.end_s + 1 <= t.start_s) | (cal.start_s >= t.end_s + 1)).all()


def test_empty_no_vacuous_pass():
    assert not technical_gates(pd.DataFrame(), pd.DataFrame())
    with pytest.raises(ValueError, match="five"):
        folds(pd.DataFrame({"event_id": range(4), "start_s": range(4)}))


def test_missing_comparator_rejected():
    rows = []
    for map_name in ("real", "population_code_permuted"):
        row = {"dataset": "test", "animal": "a", "session": "s", "event_id": 1, "split": 0, "map": map_name, "n_heldout_spikes": 0}
        row.update({"score_" + model: 0.0 for model in ("first_order_imm", "iid_position", "static_location", "diffusion", "event_global", "run_global")})
        rows.append(row)
    split, events = contrasts(pd.DataFrame(rows))
    assert split.delta.eq(0).all() and events.delta.eq(0).all()
    assert split.delta_per_heldout_spike.isna().all()
    with pytest.raises(ValueError, match="missing"):
        contrasts(pd.DataFrame(rows[:1]))


def test_known_temporal_operating_case():
    rng = np.random.default_rng(20260908)
    x = np.array(list(itertools.product(np.arange(4) * 8.0, repeat=2)))
    centers = rng.uniform(0, 24, (24, 2))
    rates = 0.01 + np.exp(-((centers[:, None] - x[None]) ** 2).sum(axis=2) / (2 * 7**2))
    train, held = np.arange(16), np.arange(16, 24)
    trp = rates[train] / rates[train].sum(axis=0)
    hep = rates[held] / rates[held].sum(axis=0)
    context = SpatialPredictionContext(x)
    true_diff = dense_kernel(x, 60 * np.sqrt(0.02))
    advantages = {"diffusion": [], "iid_position": [], "static_location": []}
    for generator, values in advantages.items():
        for _ in range(40):
            path = [int(rng.integers(len(x)))]
            for _ in range(19):
                path.append(
                    int(rng.choice(len(x), p=true_diff[:, path[-1]])) if generator == "diffusion" else int(rng.integers(len(x))) if generator == "iid_position" else path[-1]
                )
            tr = np.array([rng.multinomial(4, trp[:, j]) for j in path])
            he = np.array([rng.multinomial(3, hep[:, j]) for j in path])
            q, _ = context.infer(identity_likelihood(tr, rates[train]), np.arange(20) * 0.02)
            held_ll = identity_likelihood(he, rates[held])
            scores = {m: logsumexp(p + held_ll, axis=1).sum() for m, p in q.items()}
            comparator = "diffusion" if generator == "iid_position" else "iid_position"
            values.append(scores[generator] - scores[comparator])
    assert all(np.mean(v) > 0 for v in advantages.values())
