import itertools
import math

import numpy as np
import pytest

from scripts.kleinman_conditional_spatial_score import (
    anchor_score,
    fixed_total_moments,
    reference_feature,
    spatial_score,
)


def enumeration(c, odds, a, m):
    values, weights = [], []
    for y in itertools.product(*(range(k + 1) for k in c)):
        if sum(y) != m:
            continue
        values.append(np.dot(y, a))
        weights.append(math.prod(math.comb(int(k), int(j)) for k, j in zip(c, y, strict=True)) * np.exp(np.dot(y, odds)))
    weights = np.array(weights) / sum(weights)
    values = np.array(values)
    mean = weights @ values
    return mean, weights @ (values - mean) ** 2


@pytest.mark.parametrize("m", range(8))
def test_exact_enumeration(m):
    c, odds, a = [2, 1, 4], np.log([0.02, 10.0, 1.0]), np.array([-3.0, 1.0, 5.0])
    actual = fixed_total_moments(c, odds, a, m)
    np.testing.assert_allclose(actual, enumeration(c, odds, a, m), atol=1e-9, rtol=1e-10)


def test_gain_cancels_and_feature_constant_shifts_do_not_change_score():
    c, odds, a = [5, 7, 4], np.log([0.1, 2.0, 7]), np.array([-3.0, 0.0, 2.0])
    mu, var = fixed_total_moments(c, odds, a, 6)
    shifted, vv = fixed_total_moments(c, odds + 11, a + 10, 6)
    assert shifted == pytest.approx(mu + 60)
    assert vv == pytest.approx(var)


def test_constant_odds_known_hypergeometric_moments():
    from scipy.stats import hypergeom

    c = [20, 30]
    mu, var = fixed_total_moments(c, [0, 0], [0, 1], 17)
    assert mu == pytest.approx(hypergeom.mean(50, 30, 17))
    assert var == pytest.approx(hypergeom.var(50, 30, 17))


def test_zero_information_and_exclusive_support_are_explicit():
    s = spatial_score([0, 0, 3], [0, 5, 0], [1, 0, 1], [1, 1, 0], [-1, 2, 3])
    assert s["score"] == 0 and s["information"] == 0
    assert s["excluded_spikes"] == 8 and not s["informative"]
    s = spatial_score([3, 4], [0, 0], [1, 2], [2, 1], [-1, 1])
    assert not s["informative"]
    s = spatial_score([3, 4], [2, 1], [1, 2], [2, 1], [1, 1])
    assert not s["informative"]


@pytest.mark.parametrize("bad", [[0.5, 1], [-1, 2], [np.nan, 2]])
def test_invalid_counts_rejected(bad):
    with pytest.raises(ValueError):
        fixed_total_moments(bad, [0, 0], [1, 2], 1)


def test_impossible_spike_exposure_rejected():
    with pytest.raises(ValueError, match="exposure"):
        spatial_score([1, 0], [0, 1], [0, 1], [1, 1], [0, 1])


def test_reference_gain_and_association_intercept_invariance():
    r = np.array([0.1, 5.0, 20])
    np.testing.assert_allclose(reference_feature(r, [True] * 3), reference_feature(r * 100, [True] * 3))
    x, u, v = np.array([-1, 1, 2]), np.array([0.2, 0.5, -0.1]), np.array([2, 3, 5])
    a = anchor_score(x, u, v)
    b = anchor_score(x + 4, u + 0.7 * v, v)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_mean_null_score_zero_for_every_conditioned_table():
    c, t0, t1, f = np.array([2, 3, 1]), np.array([0.1, 0.7, 1]), np.array([0.8, 0.1, 2]), np.array([-2, 1, 4])
    m = 3
    weights, scores = [], []
    for y in itertools.product(*(range(k + 1) for k in c)):
        if sum(y) != m:
            continue
        y = np.array(y)
        weights.append(math.prod(math.comb(int(k), int(j)) for k, j in zip(c, y, strict=True)) * np.prod((t1 / t0) ** y))
        scores.append(spatial_score(c - y, y, t0, t1, f)["score"])
    assert np.average(scores, weights=weights) == pytest.approx(0, abs=1e-12)


def test_complete_synthetic_replica_preserves_all_cases_and_animals(monkeypatch):
    from scripts import calibrate_kleinman_conditional_coupling as calibration

    banks = []
    for animal in range(6):
        banks.append(
            {
                "identity": f"animal{animal}/session/test",
                "animal": f"animal{animal}",
                "session": "synthetic",
                "drug": 0,
                "novel": 0,
                "direction": 0,
                "unit_keys": np.array([[1, 1], [1, 2], [1, 3]]),
                "generating_rates": np.array([[2, 20, 5, 2], [10, 3, 12, 6], [20, 2, 4, 5]], float),
                "reference_exposure": np.full(4, 4.0),
                "baseline_exposure": np.array([1, 2, 1, 2.0]),
                "target_exposure": np.array([2, 1, 2, 1.0]),
                "predictor": np.array([-1, 0, 1.0]),
            }
        )
    monkeypatch.setattr(calibration, "BANKS", banks)
    anchors, animals, estimates = calibration.simulate_replica(0)
    assert len(anchors) == len(animals) == 6 * 7
    assert len(estimates) == 7
    assert all(r["status"] == "scored" and r["n_animals"] == 6 for r in estimates)
    assert {r["case"] for r in estimates} == set(calibration.CASES)


def test_log_partition_verifier_matches_exact_enumeration_score():
    from scripts.verify_kleinman_conditional_coupling import independent_score

    b, y = np.array([2, 4, 1]), np.array([3, 1, 2])
    t0, t1, a = np.array([0.2, 1, 0.5]), np.array([1, 0.3, 2]), np.array([-2, 1, 3.0])
    actual = independent_score(b, y, t0, t1, a)
    expected = spatial_score(b, y, t0, t1, a)
    np.testing.assert_allclose([actual["score"], actual["information"]], [expected["score"], expected["information"]], atol=2e-6, rtol=1e-6)
