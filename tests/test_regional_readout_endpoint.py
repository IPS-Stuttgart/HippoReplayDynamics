import numpy as np
import pytest

from hipporeplayimm.regional_content_frontier import mixture_fit
from hipporeplayimm.regional_readout_endpoint import (
    ContinuousDensity,
    class_likelihoods,
    count_terminal,
    fit_prevalence,
    oracle_prevalence,
)


def test_ternary_matches_existing_fitter():
    y = np.repeat([0, 1, 2, 0, 1, 2], [20, 3, 2, 1, 5, 19])
    z = np.repeat([0, 1], [25, 25])
    q = np.repeat([0, 1, 2], [60, 20, 20])
    f = class_likelihoods(y, z, q, "ternary")
    old = mixture_fit(np.array([[20, 3, 2], [1, 5, 19]]), np.array([60, 20, 20]))
    assert abs(fit_prevalence(f)-old["prevalence"]) < 1e-5


@pytest.mark.parametrize("fraction", [0., .05, .3, .8, 1.])
def test_continuous_known_separated_mixture(fraction):
    rng = np.random.default_rng(410)
    y = np.r_[rng.normal(-5, .2, 500), rng.normal(5, .2, 500)]
    z = np.repeat([0, 1], 500)
    n = round(200*fraction)
    q = np.r_[rng.normal(-5, .2, 200-n), rng.normal(5, .2, n)]
    assert abs(fit_prevalence(class_likelihoods(y, z, q, "continuous_neutral"))-fraction) < .001


def test_continuous_recovers_information_hidden_inside_neutral_category():
    rng = np.random.default_rng(41)
    y = np.r_[rng.normal(-.3, .02, 200), rng.normal(.3, .02, 200)]
    z = np.repeat([0, 1], 200)
    q = np.r_[np.full(150, -.3), np.full(50, .3)]
    assert abs(fit_prevalence(class_likelihoods(y, z, q, "continuous_neutral"))-.25) < .01
    assert np.isnan(fit_prevalence(class_likelihoods(np.ones(400), z, np.ones(200), "ternary")))


def test_no_information_is_unidentified():
    f = class_likelihoods(np.zeros(100), np.repeat([0, 1], 50), np.zeros(200), "continuous_neutral")
    assert np.isnan(fit_prevalence(f))


def test_density_is_calibration_only_and_tail_is_not_clamped():
    density = ContinuousDensity(np.linspace(-1, 1, 500))
    expected = density(np.array([0., .5]))
    np.testing.assert_array_equal(density(np.array([0., .5, 100.]))[:2], expected)
    assert density(np.array([10., 100.]))[1] < density(np.array([10., 100.]))[0]


def test_generator_oracle_handles_inverted_readouts():
    y = np.r_[np.full(100, -5.), np.full(100, 5.)]
    truth = np.repeat([0, 1], 100)
    q = np.r_[np.full(70, -5.), np.full(30, 5.)]
    normal = class_likelihoods(y, truth, q, "continuous_native")
    inverted = class_likelihoods(-y, truth, -q, "continuous_native")
    assert abs(oracle_prevalence(np.r_[normal, inverted], np.repeat(["stationary", "late_jump"], 100))-.3) < .001


@pytest.mark.parametrize("scores,truth", [([], []), ([1], [0]), ([1, np.nan], [0, 1])])
def test_reject_invalid_calibration(scores, truth):
    with pytest.raises(ValueError):
        class_likelihoods(scores, truth, np.array([1]), "continuous_native")


@pytest.mark.parametrize("f", [np.ones((0, 2)), [[1, 0]], [[1, np.nan]], [[-1, 1]]])
def test_invalid_likelihoods(f):
    with pytest.raises(ValueError):
        fit_prevalence(f)


def test_terminal_counts_preserve_spikes_and_half_open_end():
    from hipporeplayimm.selection_matched_regional import endpoint_interval
    a, b = endpoint_interval(37748.100000000006, 37748.180000000004)
    times = np.array([a, b-.01, b-.005, b-.001, b])
    template = [{"start": 37748.100000000006, "end": 37748.180000000004, "times": times}]
    identity = np.array([0, 0, 1, 1, 0])
    np.testing.assert_array_equal(count_terminal(identity, template, 2, .020), [[2, 2]])
    np.testing.assert_array_equal(count_terminal(identity, template, 2, .005), [[0, 2]])


def test_empty_short_window_is_retained():
    template = [{"start": 0., "end": .1, "times": np.array([.081])}]
    np.testing.assert_array_equal(count_terminal(np.array([0]), template, 2, .005), [[0, 0]])
