import numpy as np
import pytest

from hipporeplayimm.regional_readout_endpoint import fit_prevalence
from scripts.diagnose_regional_zero_mass import mixed_measure_likelihoods


def test_identical_zero_mass_carries_no_artificial_width_evidence():
    y = np.r_[np.zeros(600), np.linspace(-10, -1, 400), np.zeros(600), np.linspace(1, 2, 400)]
    truth = np.repeat([0, 1], 1000)
    f = mixed_measure_likelihoods(y, truth, np.array([0.]))
    assert f[0, 0] == f[0, 1]


def test_known_sparse_population_prevalence():
    rng = np.random.default_rng(89)
    y = np.r_[np.zeros(3000), rng.normal(-5, .1, 2000), np.zeros(3000), rng.normal(5, .1, 2000)]
    truth = np.repeat([0, 1], 5000)
    q = np.r_[np.zeros(120), np.full(60, -5.), np.full(20, 5.)]
    assert abs(fit_prevalence(mixed_measure_likelihoods(y, truth, q))-.25) < .005


def test_all_silent_queries_are_unidentified_if_both_classes_silent():
    f = mixed_measure_likelihoods(np.zeros(20), np.repeat([0, 1], 10), np.zeros(5))
    assert np.isnan(fit_prevalence(f))


def test_missing_nonzero_calibration_cannot_supply_continuous_density():
    with pytest.raises(ValueError):
        mixed_measure_likelihoods(np.zeros(20), np.repeat([0, 1], 10), np.ones(5))
