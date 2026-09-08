import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multinomial

from hipporeplayimm.conditional_spatial_prediction import SpatialPredictionContext, identity_likelihood
from hipporeplayimm.lagged_neural_prediction import NeuralOperator, SpatialOperator, forecasts, forward_filter, full_count_bins, mixture_scores
from scripts.audit_2d_lagged_neural_prediction import MODELS, SCORES, aggregate, decisions, score_event, validate_rows
from scripts.verify_2d_lagged_neural_prediction import reconstruct_tables, reference_scores, spatial_kernels


def neural():
    a = np.array([[0.65, 0.3, 0.05], [0.1, 0.6, 0.3], [0.3, 0.1, 0.6]])
    return NeuralOperator(np.ones(3) / 3, a, np.array([0.2, 0.3, 0.5]))


def test_dwell_control_preserves_each_self_transition():
    op = neural()
    np.testing.assert_allclose(np.diag(op.dwell), np.diag(op.transition))
    np.testing.assert_allclose(op.dwell.sum(axis=1), 1)
    assert not np.allclose(op.dwell, op.transition)
    rng = np.random.default_rng(2)
    q = rng.dirichlet(np.ones(3), 5)
    np.testing.assert_allclose(op.step(q).sum(axis=1), 1)


def test_forward_filter_matches_exact_enumeration():
    op = neural()
    ll = np.log(np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]))
    q = forward_filter(ll, op)
    for t in range(len(ll)):
        terms = np.zeros(3)
        for path in itertools.product(range(3), repeat=t + 1):
            p = op.initial[path[0]] * np.exp(ll[0, path[0]])
            for j in range(1, len(path)):
                p *= op.transition[path[j - 1], path[j]] * np.exp(ll[j, path[j]])
            terms[path[-1]] += p
        np.testing.assert_allclose(q[t], terms / terms.sum(), atol=1e-13)


@pytest.mark.parametrize("imm", [False, True])
def test_spatial_operator_dense_blocks_and_prefix_smoother(imm):
    centers = np.array([[0, 0], [4, 0], [8, 0], [4, 4]], float)
    op = SpatialOperator(centers, imm=imm)
    n = len(centers)
    for dwell in (False, True):
        blocks = []
        for dst, kernel in enumerate(op.kernels):
            k = np.ones((n, n)) / n if kernel is None else kernel.toarray()
            if dwell:
                original = k.copy()
                for src in range(n):
                    for dest in range(n):
                        k[dest, src] = original[src, src] if dest == src else (1 - original[src, src]) / (n - 1)
            blocks.append([op.mode[src, dst] * k for src in range(op.n_modes)])
        dense = np.block(blocks)
        q = np.random.default_rng(3).dirichlet(np.ones(len(op.initial)), 5)
        np.testing.assert_allclose(op.step(q, dwell=dwell), q @ dense.T, atol=1e-13)
        np.testing.assert_allclose(dense.sum(axis=0), 1)
    ll = np.random.default_rng(4).normal(0, 2, (6, n))
    filtered = forward_filter(ll, op)
    context = SpatialPredictionContext(centers)
    for t in range(len(ll)):
        post, _ = context.infer(ll[: t + 1], (np.arange(t + 1) + 0.5) * 0.02)
        expected = np.exp(post["first_order_imm" if imm else "diffusion"][-1])
        np.testing.assert_allclose(op.collapse(filtered[t : t + 1])[0], expected, atol=1e-12)


def test_forecasts_cannot_see_suffix_or_heldout_cells():
    op = neural()
    rng = np.random.default_rng(6)
    counts = rng.poisson(2, (10, 6))
    rates = rng.uniform(0.01, 1, (6, 3))

    def predict(x):
        return forecasts(identity_likelihood(x[:, :3], rates[:3]), op)

    reference = predict(counts)
    changed = counts.copy()
    changed[:, 3:] += 100
    for h in reference:
        for name, q in reference[h].items():
            np.testing.assert_array_equal(q, predict(changed)[h][name])
    changed[4:, :3] += rng.poisson(20, (6, 3))
    other = predict(changed)
    for h in reference:
        for name in ("dynamic", "dwell_only", "frozen"):
            np.testing.assert_array_equal(reference[h][name][:4], other[h][name][:4])
        np.testing.assert_array_equal(reference[h]["no_history"], other[h]["no_history"])
    assert not np.allclose(reference[2]["same_time"], other[2]["same_time"])


def test_forecast_horizon_and_no_history_clock():
    op = neural()
    ll = np.random.default_rng(9).normal(size=(7, 3))
    f = forecasts(ll, op)
    q = forward_filter(ll, op)
    for h in (1, 2, 4):
        np.testing.assert_allclose(f[h]["dynamic"], q[:-h] @ np.linalg.matrix_power(op.transition, h))
        np.testing.assert_allclose(f[h]["dwell_only"], q[:-h] @ np.linalg.matrix_power(op.dwell, h))
        for t in range(h, len(ll)):
            np.testing.assert_allclose(f[h]["no_history"][t - h], op.initial @ np.linalg.matrix_power(op.transition, t))
    assert not forecasts(ll[:1], op)


def test_mixture_is_normalized_and_zero_counts_score_zero():
    rates = np.array([[0.8, 0.2], [0.2, 0.8]])
    q = np.array([[0.3, 0.7]])
    total = 0
    for n in range(4):
        counts = np.array([[n, 3 - n]])
        ll = identity_likelihood(counts, rates)
        s = mixture_scores(q, ll)[0]
        expected = logsumexp(np.log(q[0]) + [multinomial.logpmf(counts[0], 3, p) for p in rates.T])
        assert s == pytest.approx(expected)
        total += np.exp(s)
    assert total == pytest.approx(1)
    assert mixture_scores(q, identity_likelihood(np.zeros((1, 2)), rates))[0] == pytest.approx(0)


def test_known_directed_sequence_beats_dwell_matched_control():
    n = 6
    a = 0.01 * np.ones((n, n)) / n + 0.99 * np.roll(np.eye(n), 1, axis=1)
    op = NeuralOperator(np.ones(n) / n, a, np.ones(n) / n)
    states = np.arange(36) % n
    x = 12 * np.eye(n, dtype=int)[states]
    p = 0.05 + np.eye(n)
    ll = identity_likelihood(x, p)
    f = forecasts(ll, op)
    for h in (1, 2, 4):
        correct = mixture_scores(f[h]["dynamic"], ll[h:]).sum()
        assert correct > mixture_scores(f[h]["dwell_only"], ll[h:]).sum() + 10
        assert correct > mixture_scores(f[h]["frozen"], ll[h:]).sum() + 10


def test_unstructured_transitions_have_no_route_gain():
    a = 0.7 * np.eye(4) + 0.3 / 4
    op = NeuralOperator(np.ones(4) / 4, a, np.ones(4) / 4)
    ll = np.random.default_rng(3).normal(size=(12, 4))
    for f in forecasts(ll, op).values():
        np.testing.assert_allclose(f["dynamic"], f["dwell_only"], atol=1e-14)


def test_partial_bin_and_insufficient_horizon_are_explicit():
    x, discarded = full_count_bins(np.ones((3, 2), int), [10, 10.02, 10.04, 10.047])
    assert x.shape == (2, 2) and discarded == 2
    assert set(forecasts(np.zeros((2, 3)), neural())) == {1}
    with pytest.raises(ValueError, match="partial"):
        full_count_bins(np.ones((3, 2), int), [0, 0.01, 0.03, 0.05])
    with pytest.raises(ValueError, match="horizons"):
        forecasts(np.zeros((2, 3)), neural(), (0,))


def score_fixture():
    rows = []
    for dataset, na in (("pfeiffer_foster", 4), ("tanni2022", 5)):
        for a in range(na):
            for event in range(3):
                for split in range(5):
                    for h in (1, 2, 4):
                        for model in MODELS:
                            full = 1 if event == 2 else 8
                            nt = max(full - h, 0)
                            ns = 10 if event == 0 else 0
                            scores = {c: -10.0 if ns else 0.0 if nt else np.nan for c in SCORES}
                            if ns:
                                scores["score_dynamic"] = -5.0 if model.endswith("real") else -7.0
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "animal": a,
                                    "session": str(a),
                                    "event_id": event,
                                    "split": split,
                                    "horizon": h,
                                    "model": model,
                                    "n_full_bins": full,
                                    "n_target_bins": nt,
                                    "n_heldout_target_spikes": ns,
                                    "forecast_uses_future_training": False,
                                    "forecast_uses_heldout": False,
                                    "status": "scored" if nt else "insufficient_full_bins",
                                    **scores,
                                }
                            )
    return pd.DataFrame(rows)


def test_aggregation_retains_zero_and_insufficient_events(tmp_path):
    rows = score_fixture()
    tables = aggregate(rows)
    _, events, sessions, animals, summary = tables
    assert events[events.event_id.eq(1)].delta.eq(0).all()
    assert events[events.event_id.eq(1)].delta_per_spike.isna().all()
    assert events[events.event_id.eq(2)].delta.isna().all()
    assert events[events.event_id.eq(2)].valid_neural_splits.eq(0).all()
    assert len(sessions) == len(animals) == 9 * 3 * 31
    assert len(summary) == 2 * 3 * 31 * 2
    assert decisions(summary).replicated_forecasting_lead.all()
    for name, table in zip(("splits", "events", "sessions", "animals", "summary"), tables, strict=True):
        table.to_csv(tmp_path / f"lagged_prediction_{name}.csv.gz", index=False)
    decisions(summary).to_csv(tmp_path / "lagged_prediction_decisions.csv", index=False)
    assert reconstruct_tables(tmp_path, rows) == (len(tables[0]), len(events))
    broken = summary.copy()
    broken.loc[0, "mean"] += 1
    broken.to_csv(tmp_path / "lagged_prediction_summary.csv.gz", index=False)
    with pytest.raises(AssertionError):
        reconstruct_tables(tmp_path, rows)


def test_missing_factors_and_silent_empty_scores_fail():
    x = score_fixture()
    with pytest.raises(ValueError, match="missing"):
        validate_rows(x.iloc[:-1])
    x.loc[x.event_id.eq(2), "score_dynamic"] = 0
    with pytest.raises(ValueError, match="silent empty"):
        validate_rows(x)
    with pytest.raises(ValueError, match="nonempty"):
        validate_rows(x.iloc[:0])


def test_complete_scoring_fixture_hides_heldout_from_forecasts():
    rng = np.random.default_rng(12)
    counts = rng.poisson(1, (8, 6))
    edges = np.arange(9) * 0.02
    rates = rng.uniform(0.1, 1, (6, 4))
    op = neural()
    fit = {
        "initial": op.initial,
        "transition": op.transition,
        "occupancy": np.array([0.2, 0.3, 0.5]),
        "probabilities": rng.uniform(0.1, 1, (6, 3)),
        "global_probability": np.ones(6) / 6,
    }
    spatial = {"imm": SpatialOperator(np.arange(4)[:, None] * 8), "diffusion": SpatialOperator(np.arange(4)[:, None] * 8, imm=False)}
    train, held = np.arange(3), np.arange(3, 6)
    original = score_event(counts, edges, rates, train, held, fit, spatial, np.arange(4)[::-1])
    for h in (1, 2, 4):
        independent = reference_scores(counts, rates, train, held, fit, spatial_kernels(np.arange(4)[:, None] * 8), np.arange(4)[::-1], h)
        saved = pd.DataFrame(original).query("horizon == @h").set_index("model").reindex(independent.index)
        np.testing.assert_allclose(saved[independent.columns], independent, atol=1e-12)
    counts[:, held] += 1
    changed = score_event(counts, edges, rates, train, held, fit, spatial, np.arange(4)[::-1])
    assert len(original) == len(changed) == 15
    assert all(a["forecast_sha256"] == b["forecast_sha256"] for a, b in zip(original, changed, strict=True))
    assert any(a["score_dynamic"] != b["score_dynamic"] for a, b in zip(original, changed, strict=True))
