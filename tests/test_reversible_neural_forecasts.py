import numpy as np
import pandas as pd
import pytest
from test_occupancy_matched_forecast import score_rows

from hipporeplayimm.conditional_spatial_prediction import identity_likelihood
from hipporeplayimm.frozen_posterior_prediction import posterior_sha256
from hipporeplayimm.lagged_neural_prediction import NeuralOperator, SpatialOperator, mixture_scores
from hipporeplayimm.lagged_neural_prediction import forecasts as original_forecasts
from hipporeplayimm.occupancy_matched_forecast import stationary_distribution
from hipporeplayimm.reversible_neural_forecasts import ReversibleControls, forecasts, score_model
from scripts.audit_2d_reversible_neural_forecasts import aggregate, decision
from scripts.verify_2d_lagged_neural_prediction import Reference, spatial_kernels
from scripts.verify_2d_reversible_neural_forecasts import constraints, reconstruct, scores


def check_dense(op, pi):
    control = ReversibleControls(op, pi)
    size = len(pi)
    a = op.step(np.eye(size))
    expected = (pi[:, None] * a).T / pi[:, None]
    r = control.reverse(np.eye(size))
    b = control.reversible(np.eye(size))
    np.testing.assert_allclose(r, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(b, (a + expected) / 2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(r.sum(axis=1), 1, atol=1e-10, rtol=0)
    np.testing.assert_allclose(pi @ b, pi, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.diag(b), np.diag(a), atol=1e-12, rtol=0)
    flow = pi[:, None] * a
    bf = pi[:, None] * b
    np.testing.assert_allclose(bf, bf.T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(bf + bf.T, flow + flow.T, atol=1e-12, rtol=0)
    assert (r >= 0).all() and (b >= 0).all()
    return control


@pytest.mark.parametrize("kind", ["neural", "imm", "diffusion"])
def test_constraints_and_independent_operator(kind):
    if kind == "neural":
        a = np.random.default_rng(33).dirichlet(np.ones(7), 7)
        op = NeuralOperator(np.ones(7) / 7, a, np.ones(7) / 7)
        ref = Reference(fit={"initial": op.initial, "transition": a, "occupancy": np.ones(7) / 7})
    else:
        points = np.array([[0, 0], [8, 0], [16, 0], [0, 8], [8, 8], [0, 16]], float)
        op = SpatialOperator(points, imm=kind == "imm")
        ref = Reference(spatial_kernels(points), imm=kind == "imm")
    pi, _ = stationary_distribution(op)
    control = check_dense(op, pi)
    constraints(ref, pi)
    if kind == "diffusion":
        np.testing.assert_allclose(control.reverse(np.eye(len(pi))), op.step(np.eye(len(pi))), atol=1e-11, rtol=0)


def test_directed_sequence_recovery():
    n = 5
    a = 0.92 * np.roll(np.eye(n), 1, axis=1) + 0.06 * np.eye(n) + 0.02 / n
    op = NeuralOperator(np.ones(n) / n, a, np.ones(n) / n)
    pi, _ = stationary_distribution(op)
    control = check_dense(op, pi)
    ll = identity_likelihood(10 * np.eye(n, dtype=int)[np.arange(30) % n], np.eye(n) + 0.03)
    p = forecasts(ll, op, control)
    for h, frame in p.items():
        true = mixture_scores(frame["dynamic"], ll[h:]).sum()
        for name in ("reverse", "reversible"):
            assert true > mixture_scores(frame[name], ll[h:]).sum() + 10


def test_reversible_generator_has_zero_direction_increment():
    rng = np.random.default_rng(42)
    w = rng.uniform(0.1, 1, (5, 5))
    w += w.T
    a = w / w.sum(axis=1, keepdims=True)
    pi = w.sum(axis=1) / w.sum()
    op = NeuralOperator(pi, a, pi)
    control = check_dense(op, pi)
    x, rates = rng.poisson(1, (10, 8)), rng.uniform(0.1, 1, (8, 5))
    result = score_model(x, rates, np.arange(5), np.arange(5, 8), op, control)
    for p in result.values():
        assert p["score_dynamic"] == pytest.approx(p["score_reversible"], abs=1e-12)
        assert p["score_dynamic"] == pytest.approx(p["score_reverse"], abs=1e-12)


@pytest.mark.parametrize("kind", ["neural", "imm", "diffusion"])
@pytest.mark.parametrize("permuted", [False, True])
def test_forecast_hash_reference_scores_and_heldout_isolation(kind, permuted):
    rng = np.random.default_rng(79)
    points = np.array([(8 * x, 8 * y) for x in range(5) for y in range(3)], float)
    rates = rng.lognormal(0, 2, (80, len(points)))
    x = rng.poisson(0.8, (12, 80))
    tr, he = np.arange(56), np.arange(56, 80)
    order = rng.permutation(len(points)) if permuted else np.arange(len(points))
    if kind == "neural":
        a = rng.dirichlet(np.ones(len(points)), len(points))
        op = NeuralOperator(np.ones(len(points)) / len(points), a, np.ones(len(points)) / len(points))
        ref = Reference(fit={"initial": op.initial, "transition": a, "occupancy": op.initial})
        order = None
    else:
        op = SpatialOperator(points, imm=kind == "imm")
        ref = Reference(spatial_kernels(points), imm=kind == "imm")
    pi, _ = stationary_distribution(op)
    control = ReversibleControls(op, pi)
    ll = identity_likelihood(x[:, tr], rates[tr])
    old = original_forecasts(ll if order is None else ll[:, order], op)
    actual = score_model(x, rates, tr, he, op, control, position_order=order)
    for h, p in actual.items():
        assert p["forecast_sha256"] == posterior_sha256(old[h]["dynamic"])
        expected = scores(x, rates if order is None else rates[:, order], tr, he, ref, pi, h)
        for name, v in expected.items():
            assert p[name] == pytest.approx(v, abs=1e-10)
    modified = x.copy()
    modified[:, he] += 10
    other = score_model(modified, rates, tr, he, op, control, position_order=order)
    assert all(actual[h]["forecast_sha256"] == other[h]["forecast_sha256"] for h in actual)
    p = forecasts(ll, op, control)
    ll[5:] += rng.normal(0, 5, ll[5:].shape)
    changed = forecasts(ll, op, control)
    for h in p:
        for name in p[h]:
            np.testing.assert_array_equal(p[h][name][:5], changed[h][name][:5])


def test_invalid_stationary_distribution_is_not_repaired():
    a = np.array([[0.8, 0.2], [0.4, 0.6]])
    op = NeuralOperator(np.ones(2) / 2, a, np.ones(2) / 2)
    with pytest.raises(ValueError, match="stationary"):
        ReversibleControls(op, [0.5, 0.5])
    with pytest.raises(ValueError, match="positive"):
        ReversibleControls(op, [1, 0])


def fixture_rows():
    rows = score_rows()
    good = rows.status.eq("scored")
    rows["reconstructed_forecast_hash_matches"] = good
    rows["reconstruction_error"] = np.where(good, 0.0, np.nan)
    rows["score_reverse"] = np.where(rows.n_heldout_target_spikes.gt(0), -2.0, rows.score_dynamic)
    rows["score_reversible"] = np.where(rows.n_heldout_target_spikes.gt(0), -1.5, rows.score_dynamic)
    return rows


def test_aggregation_reconstructed_and_no_substitution_of_primary(tmp_path):
    rows = fixture_rows()
    tables = aggregate(rows)
    for name, t in zip(("splits", "events", "sessions", "animals", "summary"), tables, strict=True):
        t.to_csv(tmp_path / f"reversible_forecasts_{name}.csv.gz", index=False)
    d = decision(tables[-1])
    assert d["directional_predictive_lead"]
    pd.DataFrame([d]).to_csv(tmp_path / "reversible_forecasts_decision.csv", index=False)
    assert reconstruct(tmp_path, rows) == (len(tables[0]), len(tables[1]))
    summary = tables[-1].copy()
    summary.loc[summary.contrast.eq("learned_hmm__dynamic_minus_reversible"), "ci_low"] = -1
    assert not decision(summary)["directional_predictive_lead"]
    with pytest.raises(ValueError, match="complete primary"):
        decision(summary[~summary.contrast.eq("learned_hmm__dynamic_minus_reverse")])
    summary.to_csv(tmp_path / "reversible_forecasts_summary.csv.gz", index=False)
    with pytest.raises(AssertionError):
        reconstruct(tmp_path, rows)


def test_missing_rows_and_nan_reconstructions_fail():
    rows = fixture_rows()
    with pytest.raises(ValueError, match="nonempty"):
        aggregate(rows.iloc[:0])
    with pytest.raises(ValueError, match="missing"):
        aggregate(rows.iloc[:-1])
    rows.loc[0, "reconstruction_error"] = np.nan
    with pytest.raises(ValueError, match="not reconstructed"):
        aggregate(rows)
    rows = fixture_rows()
    rows.loc[rows.status.ne("scored"), "score_reverse"] = 0
    with pytest.raises(ValueError, match="invalid reverse"):
        aggregate(rows)
