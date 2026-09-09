from itertools import product

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.physical_neural_metric import (
    STAY,
    balanced_affinity,
    expected_difference,
    forecast_identities,
    identities,
    matched_kernel,
    neural_cost,
    normalized_cost,
    oracle_screen,
    physical_cost,
    validate_kernel,
)
from scripts.screen_2d_physical_neural_metric import METRICS, VALUE, summarize


def fixture():
    centers = np.array(list(product(range(4), range(3))), dtype=float)
    rng = np.random.default_rng(572)
    rates = rng.gamma(0.7, 3, (12, 12)) + 0.05
    return centers, rates


@pytest.mark.parametrize("kind", ["physical", "neural"])
def test_matching_constraints(kind):
    centers, rates = fixture()
    cost = physical_cost(centers) if kind == "physical" else neural_cost(rates)
    k = matched_kernel(cost)
    assert np.max(abs(k.matrix.sum(axis=0) - 1)) < 1e-10
    assert np.max(abs(k.matrix - k.matrix.T)) < 1e-12
    assert np.max(abs(np.diag(k.matrix) - STAY)) < 1e-12
    assert abs(k.entropy - 0.5 * np.log(11)) < 1.1e-7
    d, scale = normalized_cost(cost)
    w = np.exp(np.maximum(-k.beta * d, -700))
    np.fill_diagonal(w, 0)
    reference = (1 - STAY) * np.exp(k.log_scale[:, None] + k.log_scale[None, :]) * w + STAY * np.eye(12)
    np.testing.assert_allclose(reference, k.matrix, atol=1e-12)
    assert scale == k.cost_scale


def test_scale_and_label_invariance():
    centers, _ = fixture()
    cost = physical_cost(centers)
    order = np.random.default_rng(53).permutation(12)
    a = matched_kernel(cost).matrix
    b = matched_kernel(cost[np.ix_(order, order)] * 72).matrix
    np.testing.assert_allclose(a[np.ix_(order, order)], b, atol=1e-10)


def test_direct_forecast_and_proper_scores():
    centers, rates = fixture()
    physical = matched_kernel(physical_cost(centers)).matrix
    neural = matched_kernel(neural_cost(rates)).matrix
    train = matched_kernel(neural_cost(rates[:8])).matrix
    held = np.arange(8, 12)
    rows = oracle_screen(rates, held, physical, neural, train)
    emissions = identities(rates[held]).T
    for row in rows:
        h = row["horizon"]
        a, n, t = [np.linalg.matrix_power(k, h) @ emissions for k in (physical, neural, train)]
        refs = [expected_difference(a, a, t), expected_difference(n, t, a), expected_difference(n, n, a), expected_difference(n, n, t)]
        for name, value in zip(METRICS, refs, strict=True):
            assert row[name] == pytest.approx(value, abs=1e-12)
        assert row[METRICS[2]] == pytest.approx(row[METRICS[1]] + row[METRICS[3]], abs=1e-12)
        assert row[METRICS[0]] >= 0
        assert row[METRICS[2]] >= 0
        assert row[METRICS[3]] >= 0


def test_identical_geometries_have_zero_contrasts():
    centers, rates = fixture()
    a = matched_kernel(physical_cost(centers)).matrix
    for row in oracle_screen(rates, np.arange(8, 12), a, a, a):
        assert max(abs(row[c]) for c in METRICS) == 0


def test_neural_geometry_uses_only_given_cells_and_is_gain_invariant():
    _, rates = fixture()
    cost = neural_cost(rates[:8])
    rates[8:] *= 100
    np.testing.assert_array_equal(cost, neural_cost(rates[:8]))
    np.testing.assert_allclose(cost, neural_cost(rates[:8] * np.arange(1, 13)), atol=1e-12)


@pytest.mark.parametrize("cost", [np.zeros((4, 4)), np.eye(4), np.full((4, 4), np.nan), -np.ones((4, 4)), np.ones((4, 3))])
def test_bad_or_uninformative_geometry_fails(cost):
    with pytest.raises(ValueError):
        matched_kernel(cost)


def test_bad_probabilities_and_nonconvergence_fail():
    with pytest.raises(ValueError):
        identities(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        expected_difference(np.array([[0.3, 0.3]]), np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]]))
    cost = physical_cost(fixture()[0])
    with pytest.raises(ValueError, match="converge"):
        balanced_affinity(cost, 2, max_iter=1)
    with pytest.raises(ValueError):
        validate_kernel(np.full((4, 4), 1 / 4))


def test_empty_and_corrupt_screen_is_not_pass():
    items = [{"dataset": d, "animal": a, "session": a + "session"} for d, a in [("pfeiffer_foster", "Rat1"), ("tanni2022", "R2470")]]
    rows = [dict(**item, split=s, horizon=h, **{m: 0.1 for m in METRICS}) for item in items for s in range(5) for h in (1, 2, 4)]
    frame = pd.DataFrame(rows)
    assert summarize(frame, items)[-1].necessary_oracle_screen_pass.iloc[0]
    frame.loc[frame.dataset.eq("tanni2022"), VALUE] = -0.1
    assert not summarize(frame, items)[-1].necessary_oracle_screen_pass.iloc[0]
    for broken in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]), frame.assign(**{VALUE: np.nan}), frame.iloc[:0]):
        with pytest.raises(ValueError):
            summarize(broken, items)


def test_forecast_is_normalized_at_all_horizons():
    centers, rates = fixture()
    a = matched_kernel(physical_cost(centers)).matrix
    for p in forecast_identities(a, rates).values():
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-9)


def test_independent_reconstruction_and_corruption_check():
    from scripts.verify_2d_physical_neural_metric import cost, kernel_from_parameters, reference_scores

    centers, rates = fixture()
    held = np.arange(8, 12)
    original_costs = [physical_cost(centers), neural_cost(rates), neural_cost(rates[:8])]
    reference_costs = [cost(centers=centers), cost(rates=rates), cost(rates=rates[:8])]
    kernels = [matched_kernel(c) for c in original_costs]
    reconstructed = []
    for original, independent, k in zip(original_costs, reference_costs, kernels, strict=True):
        np.testing.assert_allclose(original, independent, atol=1e-12)
        p = {"test__" + key: getattr(k, key) for key in ("beta", "cost_scale", "log_scale", "entropy", "target_entropy")}
        reconstructed.append(kernel_from_parameters(independent, p, "test"))
        bad = dict(p)
        bad["test__log_scale"] = p["test__log_scale"] + 0.01
        with pytest.raises(AssertionError):
            kernel_from_parameters(independent, bad, "test")
    expected = oracle_screen(rates, held, *(k.matrix for k in kernels))
    checked = reference_scores(rates, held, *reconstructed)
    for a, b in zip(expected, checked, strict=True):
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-11)
