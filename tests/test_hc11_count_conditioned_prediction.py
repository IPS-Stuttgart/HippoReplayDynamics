from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp

from hipporeplayimm.duration_occupancy import _score_first_order_imm_variable
from scripts import audit_hc11_count_conditioned_prediction as audit


def tiny_kernels(topology="linear"):
    return audit.transitions(np.array([2.0, 6.0]), np.arange(4) * 0.02, topology, 8.0)


def enumerate_paths(ll, matrices):
    n = ll.shape[1]
    paths = np.array(list(product(range(n), repeat=len(ll))))
    logp = ll[np.arange(len(ll)), paths].sum(axis=1) - np.log(n)
    for t, matrix in enumerate(matrices):
        with np.errstate(divide="ignore"):
            logp += np.log(matrix.toarray()[paths[:, t + 1], paths[:, t]])
    z = logsumexp(logp)
    post = np.array([[logsumexp(logp[paths[:, t] == x]) - z for x in range(n)] for t in range(len(ll))])
    return z, post


@pytest.mark.parametrize("model", ["diffusion", "first_order_imm"])
@pytest.mark.parametrize("topology", ["linear", "circular"])
def test_posterior_matches_enumeration(model, topology):
    ll = np.log([[0.1, 0.9], [0.6, 0.4], [0.8, 0.2]])
    kernels = tiny_kernels(topology)
    expanded = np.tile(ll, (1, 3)) if model == "first_order_imm" else ll
    z, post = enumerate_paths(expanded, kernels[model])
    if model == "first_order_imm":
        post = logsumexp(post.reshape(3, 3, 2), axis=1)
    actual_z, actual_post = audit.infer_one(ll, model, kernels)
    assert_allclose(actual_z, z, atol=1e-12)
    assert_allclose(actual_post, post, atol=1e-12)


def test_linear_imm_matches_pf_engine():
    ll = np.log([[0.1, 0.9], [0.6, 0.4], [0.8, 0.2]])
    kernels = tiny_kernels()
    expected = _score_first_order_imm_variable(
        audit.ss, ll, np.array([[2.0], [6.0]]), stationary_sigma_cm=2.0, diffusion_transitions=kernels["diffusion"], max_step_sigma=4.0, mode_stickiness=0.95
    )
    actual = audit.infer_one(ll, "first_order_imm", kernels)
    assert_allclose(actual[0], expected[0], atol=1e-12)
    assert_allclose(actual[1], expected[1], atol=1e-12)


def test_circular_wrap_and_matrix_normalization():
    centers = np.arange(2.0, 42.0, 4.0)
    circ = audit.transitions(centers, np.arange(4) * 0.02, "circular", 40.0)
    linear = audit.transitions(centers, np.arange(4) * 0.02, "linear", 40.0)
    for matrices in circ.values():
        for matrix in matrices:
            assert_allclose(np.asarray(matrix.sum(axis=0)), 1.0, atol=1e-12)
    assert circ["first_order_imm"][0][9, 0] > linear["first_order_imm"][0][9, 0]


def test_half_open_spikes_and_partial_bin():
    edges = audit.event_edges(100.0, 100.041)
    assert_allclose(np.diff(edges), [0.02, 0.02, 0.001])
    spikes = audit.native.SpikeData((1,), {1: np.array([99.99, 100.0, 100.02, 100.04, 100.041])})
    assert_allclose(audit.count_spikes(spikes, (1,), edges), [[1], [1], [1]])


@pytest.mark.parametrize("model", audit.MODELS)
def test_direction_predictive_is_normalized_and_frozen(model):
    rates = [np.array([[8.0, 1.0], [1.0, 8.0]]), np.array([[2.0, 6.0], [7.0, 1.0]])]
    counts = np.array([[1, 0], [0, 1], [1, 0]])
    parts = [audit.pf.likelihood_parts(counts, r, np.full(3, 0.02)) for r in rates]
    frozen = audit.infer_training(parts, "count_conditioned", 1.0, model, tiny_kernels())
    original = frozen.copy()
    assert_allclose(logsumexp(frozen, axis=1), 0, atol=1e-12)
    predictive = []
    for spikes in ((0, 2), (1, 1), (2, 0)):
        held = [audit.pf.likelihood_parts(np.tile(spikes, (3, 1)), r, np.full(3, 0.02)) for r in rates]
        ll = np.stack([h["count_conditioned"] for h in held]).transpose(1, 0, 2).reshape(3, 4)
        predictive.append(np.exp(logsumexp(frozen + ll, axis=1)))
        assert np.isfinite(audit.heldout_score(frozen, held, "count_conditioned"))
    assert_allclose(np.sum(predictive, axis=0), 1.0, atol=1e-12)
    assert_allclose(frozen, original, atol=0)


def test_zero_heldout_spikes_and_snapshot_permutation_invariance():
    counts = np.array([[1, 0], [0, 2], [2, 1]])
    rates = np.array([[8.0, 1.0], [1.0, 8.0]])
    for model in ("iid_position", "static_location"):
        scores = []
        for order in ([0, 1], [1, 0]):
            parts = [audit.pf.likelihood_parts(counts, rates[:, order], np.full(3, 0.02))]
            q = audit.infer_training(parts, "count_conditioned", 1.0, model, tiny_kernels())
            scores.append(audit.heldout_score(q, parts, "count_conditioned"))
            empty = [audit.pf.likelihood_parts(counts * 0, rates[:, order], np.full(3, 0.02))]
            assert_allclose(audit.heldout_score(q, empty, "count_conditioned"), 0.0, atol=1e-12)
        assert_allclose(*scores, atol=1e-12)


def selection():
    return pd.DataFrame(
        [
            {
                "session": "s",
                "animal": "rat",
                "phase": phase,
                "event_id": i,
                "match_pair_id": 1,
                "start_time_s": i,
                "end_time_s": i + 0.1,
                "scoring_time_bin_s": 0.02,
                "scoring_event_padding_s": 0,
            }
            for i, phase in enumerate(("PRE", "POST"))
        ]
    )


def fixture_scores():
    rows = []
    for e in selection().itertuples(index=False):
        for split, variant, map_name, obs, model, temp in product(range(audit.N_SPLITS), audit.VARIANTS, audit.MAPS, audit.OBSERVATIONS, audit.MODELS, audit.TEMPERATURES):
            rows.append(
                {
                    "session": e.session,
                    "rat": e.animal,
                    "phase": e.phase,
                    "event_index": e.event_id,
                    "match_pair_id": 1,
                    "split": split,
                    "encoding_variant": variant,
                    "map": map_name,
                    "observation": obs,
                    "model": model,
                    "inference_temperature": temp,
                    "heldout_temperature": 1.0,
                    "conditional_heldout_log_score": -10.0,
                    "poisson_heldout_log_score": -12.0,
                    "n_heldout_spikes": 5,
                    "posterior_unchanged": True,
                    "heldout_used_for_inference": False,
                    "train_cell_ids": "1,2,3",
                    "heldout_cell_ids": "4,5",
                    "status": "success",
                }
            )
    return pd.DataFrame(rows)


def test_selection_failures():
    audit.validate_selection(selection())
    with pytest.raises(ValueError):
        audit.validate_selection(selection(), full_cohort=True)
    with pytest.raises(ValueError):
        audit.validate_selection(selection().iloc[:0])
    with pytest.raises(ValueError):
        audit.validate_selection(pd.concat([selection(), selection()]))
    bad = selection()
    bad.loc[1, "match_pair_id"] = 2
    with pytest.raises(ValueError):
        audit.validate_selection(bad)


def test_nonvacuous_and_missing_gate_failures():
    scores = fixture_scores()
    assert audit.gate_summary(scores, selection()).passed.all()
    assert not audit.gate_summary(scores.iloc[:0], selection()).iloc[-1].passed
    assert not audit.gate_summary(scores.iloc[1:], selection()).iloc[-1].passed
    bad = scores.copy()
    bad.loc[0, "train_cell_ids"] = "1,4"
    assert not audit.gate_summary(bad, selection()).iloc[-1].passed
    bad = scores.copy()
    bad.loc[0, "heldout_temperature"] = 0.3
    assert not audit.gate_summary(bad, selection()).iloc[-1].passed
    bad = scores.copy()
    bad.loc[0, "event_index"] = 111
    assert not audit.gate_summary(bad, selection()).iloc[-1].passed


def test_event_medians_precede_aggregation():
    scores = fixture_scores()
    mask = (scores.model == "first_order_imm") & (scores["map"] == "real") & (scores.observation == "count_conditioned")
    for i, value in enumerate([1.0, 2.0, 3.0, 4.0, 500.0]):
        scores.loc[mask & scores.split.eq(i), "conditional_heldout_log_score"] += value
    split, events = audit.contrasts(scores)
    assert len(split) == 2 * 5 * 2 * 2 * 8
    assert_allclose(events.loc[events.contrast == "imm_minus_iid_position", "delta"], 3.0)
    with pytest.raises(ValueError):
        audit.contrasts(pd.concat([scores, scores.iloc[:1]]))


def test_known_local_and_moving_examples():
    centers = np.arange(6) * 4.0 + 2
    edges = np.arange(13) * 0.02
    kernels = audit.transitions(centers, edges, "linear", 24.0)
    rates = np.exp(-0.5 * ((np.arange(6)[:, None] - np.arange(6)[None, :]) / 0.45) ** 2) * 60 + 0.05
    rng = np.random.default_rng(192)
    local, moving = [], []
    for path, output in ((np.full(12, 2), local), (np.tile(np.arange(6), 2), moving)):
        for _ in range(25):
            train = rng.poisson(0.08 * rates[:, path].T)
            held = rng.poisson(0.08 * rates[:, path].T)
            tr = [audit.pf.likelihood_parts(train, rates, np.diff(edges))]
            he = [audit.pf.likelihood_parts(held, rates, np.diff(edges))]
            result = {m: audit.heldout_score(audit.infer_training(tr, "count_conditioned", 1.0, m, kernels), he, "count_conditioned") for m in audit.MODELS}
            output.append(result)
    assert np.mean([r["static_location"] - r["iid_position"] for r in local]) > 0
    assert np.mean([r["diffusion"] - r["static_location"] for r in moving]) > 0
