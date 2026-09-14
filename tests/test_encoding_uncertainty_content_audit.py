import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from scripts.audit_encoding_uncertainty_content import (
    check_metrics, exposure_from_kernel, reconstruct_uncertainty, scipy_posterior,
)
from scripts.measure_encoding_uncertainty_content import (
    effective_exposure, log_predictive, rate_uncertainty, readout,
)


def encoding():
    r = np.random.default_rng(312)
    mu = r.uniform(.01, 25, (8, 120))
    occ1, occ2 = r.uniform(.01, 1, (2, 120))
    return dict(x_edges_cm=np.arange(11)*8., y_edges_cm=np.arange(13)*8.,
        occupancy_s=occ1+occ2, occupancy_first_half_s=occ1, occupancy_second_half_s=occ2,
        rates_hz=mu, rates_first_half_hz=mu*.5, rates_second_half_hz=mu*1.5,
        valid_spatial_bins=np.ones(120, bool), unit_qc_mask=np.ones(8, bool), cell_ids=np.arange(8),
        bin_centers_cm=np.array([[x, y] for x in np.arange(10)*8+4 for y in np.arange(12)*8+4]))


def test_independent_kernel_and_uncertainty_reconstruction():
    d = encoding()
    np.testing.assert_allclose(exposure_from_kernel(d['occupancy_s'], (10, 12)),
                               effective_exposure(d['occupancy_s'], (10, 12)), atol=1e-12)
    a, b = rate_uncertainty(d), reconstruct_uncertainty(d)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize('shape', [None, .01, 5., 1000.])
def test_scipy_posterior_matches_stable_lookup(shape):
    r = np.random.default_rng(420)
    n, mu = r.poisson(.2, (40, 8)), r.uniform(.0001, 25, (8, 50))
    alpha = np.full_like(mu, shape) if shape is not None else None
    p, _ = scipy_posterior(n, mu, alpha)
    np.testing.assert_allclose(p, softmax(log_predictive(n, mu, alpha), axis=1), atol=1e-9)


def test_recomputed_metrics_detect_tampered_readout():
    d = encoding()
    u = rate_uncertainty(d)
    n = np.random.default_rng(123).poisson(.2, (20, 8))
    truth = u['grid_cm'][:20]
    frames, posteriors = readout(n, truth, u['mean_rates'], u, u['grid_cm'], [np.arange(4), np.arange(4, 8)])
    for f in frames:
        method = f.method.iloc[0]
        check_metrics(f, posteriors[0][method], posteriors[1][method], u['grid_cm'], truth)
    broken = frames[0].copy()
    broken.loc[0, 'a_brier'] += .1
    with pytest.raises(AssertionError):
        check_metrics(broken, posteriors[0]['poisson'], posteriors[1]['poisson'], u['grid_cm'], truth)


def test_real_missing_truth_is_not_converted_to_zero_error():
    d = encoding()
    u = rate_uncertainty(d)
    n = np.zeros((2, 8))
    frames, posteriors = readout(n, np.full((2, 2), np.nan), u['mean_rates'], u, u['grid_cm'], [np.arange(4), np.arange(4, 8)])
    f = frames[0]
    assert pd.isna(f.a_error).all()
    broken = f.copy()
    broken['a_error'] = 0.
    with pytest.raises(AssertionError):
        check_metrics(broken, posteriors[0]['poisson'], posteriors[1]['poisson'], u['grid_cm'], np.full((2, 2), np.nan))
