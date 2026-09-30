import numpy as np
import pytest

from hipporeplayimm.regional_terminal_mixture import (
    Calibration,
    count_subwindows,
    empirical_envelope,
    independent_any_bf,
    segment_features,
    simplex_grid,
    weighted_mixture_estimates,
)


def test_count_windows_partition_and_preserve_native_last_start():
    template = {"times": np.array([.059, .06, .079, .08, .099, .1]), "offsets": np.array([0, 6]),
                "starts": np.array([0.]), "endpoints": np.array([.1])}
    result = count_subwindows(np.array([0, 0, 1, 1, 0, 1]), template, 2, 40, np.array([.08]))
    np.testing.assert_array_equal(result, [[[1, 1], [1, 1]]])
    last = count_subwindows(np.array([0, 0, 1, 1, 0, 1]), template, 2, 20, np.array([.080000000001]))
    np.testing.assert_array_equal(last, [[[1, 0]]])


def test_short_event_is_not_padded():
    t = {"times": np.array([]), "offsets": np.array([0, 0]), "starts": np.array([.05]), "endpoints": np.array([.1])}
    with pytest.raises(ValueError, match="outside"):
        count_subwindows(np.array([], int), t, 2, 100, np.array([.08]))


def test_single_window_is_original_neutral_bf():
    np.testing.assert_array_equal(independent_any_bf([[2.], [-3.], [99.]], [[1], [2], [0]], .05), [2., -3., 0.])


def test_independent_union_corrects_increasing_prior_target():
    result = independent_any_bf(np.zeros((1, 5)), np.ones((1, 5)), .04)
    assert result[0] == pytest.approx(0., abs=1e-12)
    score = independent_any_bf([[2., -1.]], [[1, 1]], .1)[0]
    from scipy.special import expit, logit
    q = expit(np.array([2., -1.])+logit(.1))
    expected = logit(1-np.prod(1-q))-logit(1-.9**2)
    assert score == pytest.approx(expected)


def test_silent_segment_exact_atom_and_order_invariance():
    assert independent_any_bf([[7., 8.]], [[0, 0]], .1)[0] == 0.
    a = independent_any_bf([[2., -1., 5.]], [[1, 2, 1]], .1)
    b = independent_any_bf([[5., 2., -1.]], [[1, 1, 2]], .1)
    np.testing.assert_allclose(a, b)


def test_readouts_agree_at_twenty_ms():
    n = np.array([[[1, 2]], [[3, 1]], [[0, 0]]])
    f = segment_features(n, np.array([[10., 1.], [1., 10.]]), np.array([True, False]))
    np.testing.assert_array_equal(f["independent_bin_any_home"], f["pooled_counts"])


def test_zero_atom_is_not_kde_smoothed():
    cal = Calibration(np.array([0., 0., 1., 2.]), np.array([0, 0, 1, 1]), "continuous_zero_mass")
    f = cal.likelihoods(np.array([0.]))
    np.testing.assert_allclose(f, [[2.5/3, .5/3]])
    with pytest.raises(ValueError, match="support"):
        cal.likelihoods(np.array([1.]))


def test_calibration_never_accepts_missing_truth_class():
    with pytest.raises(ValueError, match="class"):
        Calibration(np.array([1., 2.]), np.array([0, 0]), "continuous_zero_mass")


def test_simplex_grid_includes_all_pures_and_210_points():
    grid = simplex_grid()
    assert grid.shape == (210, 7)
    np.testing.assert_allclose(grid.sum(axis=1), 1)
    for unit in np.eye(7):
        assert (grid == unit).all(axis=1).any()


def test_all_mixtures_lie_between_pure_mles():
    rng = np.random.default_rng(5)
    f = np.exp(rng.normal(size=(7, 24, 2)))
    envelope = empirical_envelope(f, .3)
    fit = weighted_mixture_estimates(f, simplex_grid())
    assert fit.min() >= envelope["lower_estimate"]-1e-8
    assert fit.max() <= envelope["upper_estimate"]+1e-8
    assert np.max(abs(fit-.3)) == pytest.approx(envelope["worst_absolute_error"], abs=1e-8)


def test_unidentified_pure_is_failure_not_dropped():
    f = np.ones((2, 10, 2))
    f[1, :, 1] = 2.
    e = empirical_envelope(f, .2)
    assert not e["identified"] and not e["within_5pp"]
    assert (e["lower_estimate"], e["upper_estimate"]) == (0., 1.)
    assert e["worst_absolute_error"] == .8


def test_extreme_likelihood_ratios_do_not_overflow():
    f = np.array([[[1e-300, 1.], [1., 1e-300]], [[1., 1.], [1., 1.]]])
    fit = weighted_mixture_estimates(f, [[1., 0.], [.5, .5], [0., 1.]])
    np.testing.assert_allclose(fit[:2], [.5, .5], atol=1e-10)
    assert np.isnan(fit[2])
