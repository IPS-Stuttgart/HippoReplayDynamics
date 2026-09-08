import numpy as np
import pandas as pd
import pytest
from scipy.special import entr

from hipporeplayimm.conditional_spatial_prediction import identity_likelihood
from hipporeplayimm.frozen_posterior_prediction import posterior_sha256
from hipporeplayimm.lagged_neural_prediction import NeuralOperator, SpatialOperator, forecasts, forward_filter, mixture_scores
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull, stationary_distribution
from scripts.audit_2d_lagged_neural_prediction import MODELS, SCORES
from scripts.audit_2d_lagged_neural_prediction import aggregate as original_aggregate
from scripts.audit_2d_occupancy_matched_forecasts import aggregate, decisions, score_model
from scripts.verify_2d_lagged_neural_prediction import Reference
from scripts.verify_2d_occupancy_matched_forecasts import null_forecast_scores, reconstruct, verify_parameters


def dense_reference(pi, mode, stay):
    m, n = pi.shape
    s = m * n
    fixed = np.zeros((s, s))
    allowed = np.ones((s, s))
    for i in range(s):
        for dest in range(m):
            j = dest * n + i % n
            fixed[i, j] = pi.ravel()[i] * mode[i // n, dest] * stay[dest, i % n]
            allowed[i, j] = 0
    c = np.maximum(pi.ravel() - fixed.sum(axis=0), 0)
    rows = np.array([[pi.ravel()[i] * mode[i // n, j] - fixed[i, j * n : (j + 1) * n].sum() for j in range(m)] for i in range(s)])
    for j in range(m):
        total = rows[:, j].sum()
        if total == 0:
            c[j * n : (j + 1) * n] = 0
        else:
            c[j * n : (j + 1) * n] *= total / c[j * n : (j + 1) * n].sum()
    f = allowed.copy()
    for _ in range(20000):
        for j in range(m):
            g = f[:, j * n : (j + 1) * n]
            totals = g.sum(axis=1)
            g *= np.divide(rows[:, j], totals, out=np.zeros(s), where=totals > 0)[:, None]
        totals = f.sum(axis=0)
        f *= np.divide(c, totals, out=np.zeros(s), where=totals > 0)[None, :]
        achieved = np.stack([f[:, j * n : (j + 1) * n].sum(axis=1) for j in range(m)], axis=1)
        if np.max(np.abs(achieved - rows)) < 1e-13:
            return (fixed + f) / pi.ravel()[:, None]
    raise AssertionError("dense reference scaling failed")


def assert_constraints(op, null):
    s = len(op.initial)
    original = op.step(np.eye(s))
    b = null.step(np.eye(s))
    pi = null.pi.ravel()
    np.testing.assert_allclose(pi @ original, pi, atol=1e-12)
    np.testing.assert_allclose(pi @ b, pi, atol=1e-11)
    np.testing.assert_allclose(b.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(np.diag(b), np.diag(original), atol=1e-12)
    m, n = null.pi.shape
    for i in range(s):
        for dest in range(m):
            np.testing.assert_allclose(b[i, dest * n : (dest + 1) * n].sum(), null.mode[i // n, dest], atol=1e-12)
            assert b[i, dest * n + i % n] == pytest.approx(original[i, dest * n + i % n], abs=1e-12)
    assert (b >= 0).all()
    np.testing.assert_allclose(b, dense_reference(null.pi, null.mode, null.stay), atol=1e-10)
    assert (pi[:, None] * entr(b)).sum() >= (pi[:, None] * entr(original)).sum() - 1e-10


def test_neural_equilibrium_is_not_empirical_occupancy():
    a = np.array([[0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.1, 0.1, 0.8]])
    op = NeuralOperator(np.ones(3) / 3, a, np.array([0.1, 0.2, 0.7]))
    null = OccupancyMatchedNull.from_operator(op)
    assert not np.allclose(null.pi.ravel(), op.initial)
    assert_constraints(op, null)
    assert not np.allclose(null.step(np.eye(3)), op.dwell)


@pytest.mark.parametrize("imm", [False, True])
def test_spatial_null_preserves_modes_dwell_and_joint_occupancy(imm):
    points = np.array([[0, 0], [8, 0], [16, 0], [0, 8], [8, 8], [0, 16]], float)
    op = SpatialOperator(points, imm=imm)
    null = OccupancyMatchedNull.from_operator(op)
    assert_constraints(op, null)
    assert null.equilibrium_error < 1e-11 and null.mode_error < 1e-10
    if imm:
        assert np.all(null.stay[0] == 1)
        assert np.all(null.u[:, 0] == 0)


def test_directed_sequence_beats_matched_null():
    n = 5
    a = 0.92 * np.roll(np.eye(n), 1, axis=1) + 0.06 * np.eye(n) + 0.02 / n
    op = NeuralOperator(np.ones(n) / n, a, np.ones(n) / n)
    null = OccupancyMatchedNull.from_operator(op)
    x = 10 * np.eye(n, dtype=int)[np.arange(30) % n]
    ll = identity_likelihood(x, 0.05 + np.eye(n))
    f = forward_filter(ll, op)
    dynamic, dwell = f.copy(), f.copy()
    for h in range(1, 5):
        dynamic, dwell = op.step(dynamic), null.step(dwell)
        if h in (1, 2, 4):
            assert mixture_scores(dynamic[:-h], ll[h:]).sum() > mixture_scores(dwell[:-h], ll[h:]).sum() + 10


def test_maximum_entropy_generator_has_zero_contrast():
    pi = np.array([0.1, 0.2, 0.3, 0.4])
    null = OccupancyMatchedNull(pi[None, :], np.ones((1, 1)), np.array([[0.3, 0.5, 0.6, 0.7]]))
    a = null.step(np.eye(4))
    op = NeuralOperator(np.ones(4) / 4, a, pi)
    second = OccupancyMatchedNull.from_operator(op)
    q = np.random.default_rng(1).dirichlet(np.ones(4), 10)
    np.testing.assert_allclose(op.step(q), second.step(q), atol=1e-11)


def test_future_and_heldout_mutations_do_not_change_origin_forecast():
    rng = np.random.default_rng(5)
    a = rng.dirichlet(np.ones(4), 4)
    op = NeuralOperator(np.ones(4) / 4, a, np.ones(4) / 4)
    null = OccupancyMatchedNull.from_operator(op)
    counts = rng.poisson(1, (10, 6))
    maps = rng.uniform(0.1, 1, (6, 4))

    def predict(c):
        f = forward_filter(identity_likelihood(c[:, :3], maps[:3]), op)
        return null.step(null.step(f))

    original = predict(counts)
    counts[:, 3:] += 30
    np.testing.assert_array_equal(original, predict(counts))
    counts[4:, :3] += rng.poisson(10, (6, 3))
    np.testing.assert_array_equal(original[:4], predict(counts)[:4])


def test_inconsistent_stationary_mode_mass_is_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        OccupancyMatchedNull(np.array([[0.4, 0.4], [0.1, 0.1]]), np.array([[0.5, 0.5], [0.5, 0.5]]), np.full((2, 2), 0.8))
    with pytest.raises(ValueError, match="converge"):
        OccupancyMatchedNull(np.ones((1, 3)) / 3, np.ones((1, 1)), np.full((1, 3), 0.5), max_iter=0)


def test_stationary_distribution_matches_power_limit():
    a = np.array([[0.8, 0.1, 0.1], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]])
    op = NeuralOperator(np.ones(3) / 3, a, np.ones(3) / 3)
    pi, _ = stationary_distribution(op)
    np.testing.assert_allclose(pi, op.initial @ np.linalg.matrix_power(a, 1000), atol=1e-12)


def test_producer_scores_match_independent_reference_and_hide_targets():
    rng = np.random.default_rng(22)
    a = rng.dirichlet(np.ones(4), 4)
    fit = {"initial": np.ones(4) / 4, "transition": a, "occupancy": np.ones(4) / 4}
    op = NeuralOperator(fit["initial"], a, fit["occupancy"])
    null = OccupancyMatchedNull.from_operator(op)
    x = rng.poisson(1, (9, 6))
    rates = rng.uniform(0.1, 1, (6, 4))
    train, held = np.arange(3), np.arange(3, 6)
    result = score_model(x, rates, train, held, op, null)
    verify_parameters(null.parameters(), Reference(fit=fit))
    for h, row in result.items():
        expected = null_forecast_scores(x, rates, train, held, Reference(fit=fit), null.parameters(), h)
        assert row["score_matched"] == pytest.approx(expected, abs=1e-12)
    x[:, held] += 10
    changed = score_model(x, rates, train, held, op, null)
    assert all(result[h]["dynamic_forecast_sha256"] == changed[h]["dynamic_forecast_sha256"] for h in result)
    broken = null.parameters()
    broken["u"] = broken["u"].copy()
    broken["u"][0] *= 1.1
    with pytest.raises(AssertionError):
        verify_parameters(broken, Reference(fit=fit))


@pytest.mark.parametrize("imm", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_spatial_forecasts_reproduce_original_likelihood_order(imm, permuted):
    rng = np.random.default_rng(71)
    points = np.array([(8 * x, 8 * y) for x in range(7) for y in range(5)], float)
    rates = rng.lognormal(0, 2, (80, len(points)))
    counts = rng.poisson(0.8, (13, 80))
    train, held = np.arange(56), np.arange(56, 80)
    order = rng.permutation(len(points)) if permuted else np.arange(len(points))
    op = SpatialOperator(points, imm=imm)
    null = OccupancyMatchedNull.from_operator(op)
    expected = forecasts(identity_likelihood(counts[:, train], rates[train])[:, order], op)
    target = identity_likelihood(counts[:, held], rates[held])
    if permuted:
        target = target[:, order]
    actual = score_model(counts, rates, train, held, op, null, position_order=order)
    for h, prediction in expected.items():
        assert actual[h]["dynamic_forecast_sha256"] == posterior_sha256(prediction["dynamic"])
        score = mixture_scores(prediction["dynamic"], target[h:]).sum()
        assert actual[h]["score_dynamic"] == pytest.approx(score, abs=1e-12)
    with pytest.raises(ValueError, match="position permutation"):
        score_model(counts, rates, train, held, op, null, position_order=np.zeros(len(points), int))


def score_rows():
    rows = []
    for dataset, na in (("pfeiffer_foster", 4), ("tanni2022", 5)):
        for animal in range(na):
            for event in range(3):
                for split in range(5):
                    for h in (1, 2, 4):
                        for model in MODELS:
                            full = 1 if event == 2 else 8
                            nt = max(full - h, 0)
                            ns = 12 if event == 0 else 0
                            value = -3.0 if ns else 0.0 if nt else np.nan
                            record = {
                                "dataset": dataset,
                                "animal": f"A{animal}",
                                "session": f"S{animal}",
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
                                "original_forecast_hash_matches": bool(nt),
                                "original_score_reconstruction_error": 0.0 if nt else np.nan,
                                **{name: value for name in SCORES},
                                "score_occupancy_dwell": -2.0 if ns else value,
                            }
                            if ns:
                                record["score_dynamic"] = -1.0
                            rows.append(record)
    return pd.DataFrame(rows)


def test_complete_aggregation_and_independent_reconstruction(tmp_path):
    rows = score_rows()
    tables = aggregate(rows)
    original = original_aggregate(rows)[-1]
    events = tables[1]
    assert events[events.event_id.eq(1)].delta.eq(0).all()
    assert events[events.event_id.eq(1)].delta_per_spike.isna().all()
    assert events[events.event_id.eq(2)].delta.isna().all()
    assert events[events.event_id.eq(2)].valid_neural_splits.eq(0).all()
    for name, t in zip(("splits", "events", "sessions", "animals", "summary"), tables, strict=True):
        t.to_csv(tmp_path / f"occupancy_matched_{name}.csv.gz", index=False)
    decision = decisions(tables[-1], original)
    assert decision.all_updated_controls_pass.all() and not decision.original_compound_verdict_changed.any()
    decision.to_csv(tmp_path / "occupancy_matched_decisions.csv", index=False)
    assert reconstruct(tmp_path, rows, original) == (len(tables[0]), len(events))
    broken = tables[-1].copy()
    broken.loc[0, "mean"] += 1
    broken.to_csv(tmp_path / "occupancy_matched_summary.csv.gz", index=False)
    with pytest.raises(AssertionError):
        reconstruct(tmp_path, rows, original)


def test_incomplete_or_silent_empty_controls_fail():
    rows = score_rows()
    with pytest.raises(ValueError, match="missing"):
        aggregate(rows.iloc[:-1])
    with pytest.raises(ValueError, match="nonempty"):
        aggregate(rows.iloc[:0])
    rows.loc[rows.event_id.eq(2), "score_occupancy_dwell"] = 0
    with pytest.raises(ValueError, match="invalid matched"):
        aggregate(rows)
    rows = score_rows()
    rows.loc[rows.event_id.eq(0), "original_score_reconstruction_error"] = np.nan
    with pytest.raises(ValueError, match="not reconstructed"):
        aggregate(rows)
