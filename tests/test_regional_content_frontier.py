import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.regional_content_frontier import (
    block_histograms,
    bootstrap_compatibility,
    discrimination,
    generate_counts,
    mixture_fit,
    regional_log_bf,
    thin_counts,
)


def test_region_bf_integrates_likelihood_not_cell_bfs():
    rates = np.array([[2., 10., 5.], [8., 1., 3.]])
    n = np.array([[0, 0], [1, 2]])
    region = np.array([True, False, False])
    ll = n @ np.log(rates)-.02*rates.sum(axis=0)
    expected = ll[:, 0]-logsumexp(ll[:, 1:], axis=1)+np.log(2)
    np.testing.assert_allclose(regional_log_bf(n, rates, region), expected)


def test_conditional_likelihood_removes_position_total_rate():
    rates = np.array([[2., 20.], [3., 30.]])
    n = np.array([[1, 0], [0, 0], [3, 2]])
    region = np.array([True, False])
    np.testing.assert_allclose(regional_log_bf(n, rates, region, conditional=True), 0, atol=1e-12)
    assert np.max(np.abs(regional_log_bf(n, rates, region))) > 1


def test_correct_exposure_scales_poisson_silence():
    rates = np.array([[1., 10.]])
    score = regional_log_bf(np.zeros((2, 1), int), rates, [True, False], exposure=[.02, 0])
    np.testing.assert_allclose(score, [.18, 0])


def test_ties_and_uninformative_scores():
    m = discrimination(np.zeros(20), np.repeat([0, 1], 10))
    assert m["auc"] == .5
    assert m["balanced_error"] == .5
    assert m["balanced_brier"] == .25
    assert m["balanced_log_loss"] == pytest.approx(np.log(2))


def test_separation_and_error_not_interchangeable():
    m = discrimination(np.array([1., 2., 3., 4.]), np.array([0, 0, 1, 1]))
    assert m["auc"] == 1
    assert m["balanced_error"] == .5


def test_binomial_thinning_cannot_create_spikes():
    n = np.array([[0, 0], [1, 2], [100, 100], [4, 3]])
    out, p, unavailable = thin_counts(n, [5, 10, 20, 0], np.random.default_rng(7))
    assert np.all(out <= n)
    np.testing.assert_allclose(p, [0, 1, .1, 0])
    np.testing.assert_array_equal(unavailable, [True, True, False, False])
    np.testing.assert_array_equal(out[1], n[1])
    np.testing.assert_array_equal(out[[0, 3]], 0)


def test_mixture_fit_recovers_known_call_prevalence():
    cal = np.array([[900, 100, 0], [0, 100, 900]])
    target = .7*cal[0]+.3*cal[1]
    assert mixture_fit(cal, target)["prevalence"] == pytest.approx(.3, abs=.002)
    assert mixture_fit(np.ones((2, 3)), [1, 2, 3])["status"] == "unidentified"
    assert mixture_fit(np.zeros((2, 3)), [1, 2, 3])["status"] == "missing_class"


def test_blocks_preserve_category_dependence():
    y = np.array([0, 2, 0, 1])
    blocks = np.array([3, 3, 7, 7])
    z = np.array([0, 1, 0, 1])
    h = block_histograms(y, blocks, z)
    assert h.shape == (2, 2, 3)
    np.testing.assert_array_equal(h.sum(axis=0), [[2, 0, 0], [0, 1, 1]])


def test_compatibility_needs_calibration_and_widens_with_slack():
    c = np.tile(np.array([[[9, 1, 0], [0, 1, 9]]]), (20, 1, 1))
    o = np.tile(np.array([[6, 1, 3]]), (20, 1))
    _, rows, _ = bootstrap_compatibility(c, o, np.random.default_rng(1), 30)
    assert rows[0]["lower"] == pytest.approx(1/3, abs=1e-7)
    assert rows[-1]["lower"] == 0
    assert rows[-1]["upper"] == 1
    width = [r["upper"]-r["lower"] for r in rows]
    assert np.all(np.diff(width) >= -1e-8)
    c[:, 1] = 0
    _, rows, _ = bootstrap_compatibility(c, o, np.random.default_rng(1), 30)
    assert all(r["lower"] == 0 and r["upper"] == 1 for r in rows)


def test_multinomial_generation_preserves_sampled_total():
    rates = np.array([[1., 9.], [9., 1.]])
    args = (rates, np.array([True, False]), np.array([0, 1, 3]), 100)
    a = generate_counts(*args, np.random.default_rng(13), "conditional_multinomial", 4)
    b = generate_counts(*args, np.random.default_rng(13), "conditional_multinomial", 4)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a[0].sum(axis=1), a[3])
    assert set(a[3]).issubset({0, 4, 12})


def test_genuine_poisson_model_separates_regions():
    rates = np.array([[2., 80.], [80., 2.]])
    n, z, _, _ = generate_counts(rates, np.array([True, False]), [1], 2000,
                                  np.random.default_rng(11), "poisson", gain=2)
    bf = regional_log_bf(n, rates, [True, False], exposure=.04)
    m = discrimination(bf, z)
    assert m["auc"] > .95
    assert m["balanced_error"] < .1


def test_invalid_count_and_region_rejected():
    with pytest.raises(ValueError):
        regional_log_bf([[.5]], [[1., 2.]], [True, False])
    with pytest.raises(ValueError):
        regional_log_bf([[1]], [[1., 2.]], [True, True])
