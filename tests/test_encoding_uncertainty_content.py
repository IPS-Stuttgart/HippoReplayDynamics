import numpy as np
import pytest
from scipy.special import softmax
from scipy.stats import nbinom

from scripts.measure_encoding_uncertainty_content import (
    effective_exposure, endpoint_arrays, entropy, log_predictive,
    match_entropy, posterior_metrics, rate_uncertainty, readout,
)


def tiny_encoding():
    r = np.random.default_rng(200)
    mu = r.uniform(.02, 10, (10, 25))
    return dict(valid_spatial_bins=np.ones(25, bool), unit_qc_mask=np.ones(10, bool),
        x_edges_cm=np.arange(6)*8., y_edges_cm=np.arange(6)*8.,
        bin_centers_cm=np.array([[x, y] for x in np.arange(5)*8+4 for y in np.arange(5)*8+4]),
        occupancy_s=np.ones(25)*4, occupancy_first_half_s=np.ones(25)*2,
        occupancy_second_half_s=np.ones(25)*2, rates_hz=mu,
        rates_first_half_hz=mu*.7, rates_second_half_hz=mu*1.3, cell_ids=np.arange(10))


def test_gamma_predictive_matches_independent_scipy_pmf():
    r = np.random.default_rng(20)
    n, mu, alpha = r.poisson(3, (30, 8)), r.uniform(.001, 50, (8, 9)), r.uniform(.02, 30, (8, 9))
    expected = np.sum(nbinom.logpmf(n[:, :, None], alpha[None], alpha[None]/(alpha[None]+.02*mu[None])), axis=1)
    np.testing.assert_allclose(log_predictive(n, mu, alpha), expected, atol=1e-10, rtol=1e-10)


def test_large_shape_poisson_limit_and_zero_counts():
    r = np.random.default_rng(21)
    n, mu = r.poisson(1, (20, 5)), r.uniform(.001, 50, (5, 9))
    n[0] = 0
    np.testing.assert_allclose(log_predictive(n, mu, np.full_like(mu, 1e12)), log_predictive(n, mu), atol=1e-9)
    assert np.isfinite(log_predictive(n, mu, np.full_like(mu, 1e-7))).all()


@pytest.mark.parametrize("change", ["negative", "fractional", "nan", "bad_shape"])
def test_invalid_observations_fail(change):
    n, mu = np.zeros((2, 3)), np.ones((3, 5))
    if change == "negative": n[0, 0] = -1
    if change == "fractional": n[0, 0] = .5
    if change == "nan": n[0, 0] = np.nan
    if change == "bad_shape": mu = np.ones((2, 5))
    with pytest.raises(ValueError):
        log_predictive(n, mu)


def test_effective_exposure_matches_explicit_spatial_kernel():
    r = np.random.default_rng(30)
    shape = (7, 8)
    coords = np.array([(x, y) for x in range(shape[0]) for y in range(shape[1])])
    delta = np.abs(coords[:, None]-coords[None])
    one_d = np.exp(-.5*(np.arange(-6, 7)/1.5)**2)
    one_d /= one_d.sum()
    w = np.prod(one_d[np.minimum(delta, 6)+6], axis=2)*(delta <= 6).all(axis=2)
    occ = r.uniform(0, 8, len(coords))
    s, q = w @ occ, (w*w) @ occ
    expected = s*np.maximum(s, .05)/q
    np.testing.assert_allclose(effective_exposure(occ, shape), expected, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(effective_exposure(occ, shape, sigma=0), np.maximum(occ, .05))


def test_uncertainty_uses_no_heldout_or_replay_data_and_preserves_means():
    data = tiny_encoding()
    a = rate_uncertainty(data)
    np.testing.assert_array_equal(a["mean_rates"], data["rates_hz"])
    data.update(spikes=np.random.default_rng(1).normal(size=(100, 2)), posterior=np.zeros((20, 25)), heldout_rates=100000.)
    b = rate_uncertainty(data)
    for key in a: np.testing.assert_array_equal(a[key], b[key])
    assert (a["drift_shape"] <= a["exposure_shape"]).all()
    data["occupancy_first_half_s"][:] = 0
    assert (rate_uncertainty(data)["temporal_variance"] == 0).all()


def test_entropy_control_matches_targets_and_identifies_unreachable_ties():
    r = np.random.default_rng(22)
    logits = r.normal(0, 4, (20, 40))
    for temperature in (.15, 1, 5):
        target = entropy(softmax(logits/temperature, axis=1))
        p, log_t, good = match_entropy(logits, target)
        assert good.all()
        np.testing.assert_allclose(entropy(p), target, atol=1e-10)
        np.testing.assert_allclose(np.exp(log_t), temperature, atol=1e-6)
    _, _, good = match_entropy(np.zeros((1, 10)), np.array([.5]))
    assert not good[0]


def test_fixed_endpoint_and_truth_are_unchanged():
    source = dict(event_ids=np.array([9, 18]), offsets=np.array([0, 10, 18]),
        counts=np.arange(18*10).reshape(18, 10), starts_s=np.array([1., 7.]),
        truth_base_cm=np.arange(36).reshape(18, 2))
    n, truth, start = endpoint_arrays(source)
    np.testing.assert_array_equal(n[0], source["counts"][6:10].sum(0))
    np.testing.assert_array_equal(truth[1], source["truth_base_cm"][14:18].mean(0))
    np.testing.assert_allclose(start, [1.03, 7.02])


def test_b_spikes_cannot_affect_a_posterior_diagnostic_or_temperature():
    data = tiny_encoding()
    u = rate_uncertainty(data)
    counts = np.random.default_rng(33).poisson(.5, (12, 10))
    groups = [np.arange(5), np.arange(5, 10)]
    args = (np.full((12, 2), np.nan), u["mean_rates"], u, u["grid_cm"], groups)
    before, pb = readout(counts, *args)
    counts[:, 5:] += 10
    after, pa = readout(counts, *args)
    for name in pb[0]: np.testing.assert_array_equal(pb[0][name], pa[0][name])
    for a, b in zip(before, after, strict=True):
        for key in ("a_encoding_sensitivity_tv", "a_matched_log_temperature", "a_entropy", "a_x_cm"):
            np.testing.assert_array_equal(a[key], b[key])


def test_uniform_posterior_does_not_pass_proper_localization_scores():
    grid = np.array([[x, y] for x in (0, 8, 16) for y in (0, 8, 16)], dtype=float)
    truth = grid[:1]
    flat = posterior_metrics(np.full((1, 9), 1/9), grid, truth)
    localized = posterior_metrics(np.eye(9)[:1], grid, truth)
    assert flat["brier"][0] > localized["brier"][0]
    assert flat["error"][0] > localized["error"][0]
    assert flat["nll"][0] > localized["nll"][0]
