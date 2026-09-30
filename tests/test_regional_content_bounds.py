import numpy as np
import pytest

from hipporeplayimm.regional_content_bounds import (
    binomial_interval,
    calibration_intervals,
    coverage_groups,
    decode_region,
    fit_latent_class,
    identification_set,
    observed_intervals,
    pattern_counts,
    patterns,
)


def perfect_case():
    table = patterns(3)
    p = np.zeros(len(table))
    p[0], p[-1] = .7, .3
    theta = np.zeros((2, 3, 3))
    theta[0, :, 0], theta[1, :, 2] = 1, 1
    return np.stack([p, p], axis=-1), np.stack([theta, theta], axis=-1)


def test_exact_perfect_bounds_and_conditionals():
    obs, cal = perfect_case()
    model = identification_set(obs, cal)
    assert model.prevalence()["lower"] == pytest.approx(.3)
    assert model.prevalence()["upper"] == pytest.approx(.3)
    assert model.conditional(0)["upper"] == pytest.approx(0)
    assert model.conditional(26)["lower"] == pytest.approx(1)


def test_no_assumptions_no_identification():
    obs, cal = perfect_case()
    model = identification_set(obs, cal, slack=1)
    assert model.prevalence()["lower"] == pytest.approx(0)
    assert model.prevalence()["upper"] == pytest.approx(1)
    assert model.conditional(26)["lower"] == pytest.approx(0)
    assert model.conditional(26)["upper"] == pytest.approx(1)


def test_widening_is_monotone():
    obs, cal = perfect_case()
    result = [identification_set(obs, cal, s).prevalence() for s in (0, .05, .1, .2, 1)]
    assert np.all(np.diff([r["lower"] for r in result]) <= 1e-9)
    assert np.all(np.diff([r["upper"] for r in result]) >= -1e-9)


def test_correlated_errors_are_permitted():
    # All views make the same error, conditionally perfectly dependent.
    theta = np.array([[[.8, 0, .2]]*3, [[.1, 0, .9]]*3])
    p = np.zeros(27)
    p[0], p[-1] = .59, .41
    model = identification_set(np.stack([p, p], -1), np.stack([theta, theta], -1))
    result = model.prevalence()
    assert result["lower"] == pytest.approx(.3)
    assert result["upper"] == pytest.approx(.3)


def test_contradictory_constraints_not_silent_pass():
    obs, cal = perfect_case()
    obs[:] = 0
    obs[1] = 1
    assert identification_set(obs, cal).prevalence()["status"] == "incompatible"


def test_missing_truth_class_is_unknown():
    y = np.zeros((20, 4), int)
    cal, _, totals = calibration_intervals(y, np.zeros(20))
    assert totals[1] == 0
    assert np.all(cal[1, ..., 0] == 0)
    assert np.all(cal[1, ..., 1] == 1)


def test_binomial_empty_and_extremes():
    ci = binomial_interval([0, 0, 10], [0, 10, 10], .05)
    np.testing.assert_array_equal(ci[0], [0, 1])
    assert ci[1, 0] == 0 and ci[2, 1] == 1
    with pytest.raises(ValueError):
        binomial_interval(11, 10, .05)


def test_odds_and_silence_preserved():
    rates = np.array([[10., 1., 1.], [1., 10., 10.]])
    counts = np.array([[1, 0], [0, 0]])
    result = decode_region(counts, rates, [True, False, False], [np.array([0]), np.array([1])])
    q = result["mass"]
    np.testing.assert_allclose(result["log_bf"], np.log(q/(1-q))-np.log(.5))
    assert np.all(result["calls"][1] == 1)
    assert not np.allclose(q[1], 1/3)


def test_groups_disjoint_and_coverage_sorted():
    rates = np.arange(1, 22).reshape(7, 3)
    groups, score = coverage_groups(rates, [True, False, False], np.arange(7))
    assert len(np.unique(np.concatenate(groups))) == 7
    assert max(map(len, groups))-min(map(len, groups)) <= 1
    assert np.all(np.diff(score[np.concatenate(groups)]) >= 0)
    with pytest.raises(ValueError):
        decode_region(np.ones((2, 7)), rates, [True, False, False], [np.array([0, 1]), np.array([1, 2])])


@pytest.mark.parametrize("bad", [np.array([[3, 0]]), np.empty((0, 4)), np.array([[np.nan, 1]])])
def test_bad_readouts(bad):
    with pytest.raises(ValueError):
        pattern_counts(bad)


def test_pattern_encoding_and_intervals():
    y = patterns(4)
    np.testing.assert_array_equal(pattern_counts(y), np.ones(81))
    ci = observed_intervals(y)
    assert np.all(ci[:, 0] < 1/81) and np.all(ci[:, 1] > 1/81)


def test_latent_identifies_clear_patterns_but_not_their_meaning():
    y = np.r_[np.zeros((140, 4), int), np.full((60, 4), 2)]
    fit = fit_latent_class(y, seed=13)
    assert fit["pi"] == pytest.approx(.3, abs=1e-5)
    assert fit["event_probability"][:140].max() < 1e-5
    assert fit["event_probability"][140:].min() > .999
    # Inference has no truth argument; shared activity could produce these calls.


def test_same_sensitivity_and_false_alarm_no_prevalence_information():
    p = np.zeros(27)
    p[0], p[-1] = .2, .8
    theta = np.tile([.2, 0, .8], (2, 3, 1))
    result = identification_set(np.stack([p, p], -1), np.stack([theta, theta], -1)).prevalence()
    assert result["lower"] == pytest.approx(0)
    assert result["upper"] == pytest.approx(1)
