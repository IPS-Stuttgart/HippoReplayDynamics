import numpy as np
import pytest

from hipporeplayimm.regional_occupancy import (
    bin_masses,
    correct_occupancy,
    emission_means,
    invariance,
    window_occupancy,
)


def test_stationary_and_half_window_crossing():
    grid = np.array([[0., 0.], [40., 0.]])
    np.testing.assert_allclose(window_occupancy(grid, [0, .1], [0, 0], "stationary", [0, 0], 100), 1)
    np.testing.assert_allclose(window_occupancy(grid, [0, .02], [0, 1], "moving", [0, 0], 20), [.5])


def test_jumps_do_not_interpolate_and_bins_are_chronological():
    grid = np.array([[0., 0.], [100., 0.]])
    np.testing.assert_allclose(window_occupancy(grid, [0, .01, .04], [0, 1, 1], "jumping", [0, 0], 40), [0, .5])


def test_incomplete_path_fails():
    with pytest.raises(ValueError):
        window_occupancy(np.zeros((1, 2)), [0, .02], [0, 0], "stationary", [0, 0], 40)


def test_silence_and_direct_normalized_likelihood():
    rates = np.array([[30., 1., 2.], [2., 8., 10.]])
    counts = np.array([[[0, 0], [2, 1]]])
    masses = bin_masses(counts, rates, np.array([True, False, False]))
    ll = counts[0]@np.log(rates)-.02*rates.sum(axis=0)
    probs = np.exp(ll-ll.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(masses["poisson_silence"][0], probs[:, 0])
    assert masses["neutral_silence"][0, 0] == pytest.approx(1/3)


def test_time_identity_is_algebra_not_validation():
    o, q = np.array([0, .2, .8, 1]), np.array([.1, .7, .3, .9])
    e = emission_means(q, o)
    assert q.mean() == pytest.approx(o.mean()*e["s"]+(1-o.mean())*e["f"])


def test_perfect_fractional_readout_can_have_noninvariant_emissions():
    a, b = np.array([0., 1.]), np.array([.2, .8])
    ea, eb = emission_means(a, a), emission_means(b, b)
    assert not invariance([ea["s"], eb["s"]], [ea["f"], eb["f"]], expected=2)["invariance_pass"]
    assert a.mean() == b.mean() == .5


def test_pure_window_missing_support_not_dropped():
    e = emission_means([.2, .8], [.2, .8], "pure_window")
    assert not e["support_complete"]
    assert not invariance([np.nan]*7, [0.]*7)["invariance_pass"]
    assert not invariance([.9]*6, [.1]*6)["invariance_pass"]
    assert invariance([.9]*7, [.1]*7)["invariance_pass"]


def test_correction_not_clipped_and_uninformative_fails():
    assert correct_occupancy(.9, .6, .2) > 1
    assert np.isnan(correct_occupancy(.5, .2, .2))
    assert np.isnan(correct_occupancy(.5, .1, .2))


def test_calibration_transfer_for_known_linear_response():
    o = np.array([0., 0., 1., 1.])
    e = emission_means(.1+.7*o, o)
    assert correct_occupancy(.1+.7*.05, e["s"], e["f"]) == pytest.approx(.05)
    assert invariance([.8]*7, [.1]*6+[.15])["invariance_pass"] is False
