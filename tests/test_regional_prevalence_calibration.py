import numpy as np
import pytest

from scripts.measure_regional_prevalence_calibration import (
    adjust, fit_response, native_bank, regional_scores, synthetic_bank,
)


def test_affine_response_recovers_nonuniform_prevalence():
    labels = np.repeat([False, True], 100)
    fit = fit_response(np.where(labels, .7, .1), labels)
    for pi in (.05, .15, .3, .5, .75):
        estimate, status = adjust(.1 + .6 * pi, fit)
        assert status == 'available'
        assert estimate == pytest.approx(pi)


def test_constant_response_cannot_claim_agreement():
    fit = fit_response(np.full(200, .3), np.repeat([False, True], 100))
    estimate, status = adjust(.3, fit)
    assert np.isnan(estimate)
    assert status == 'weak_response_gap'


def test_incompatible_unclipped_and_missing_class():
    fit = fit_response(np.repeat([.1, .4], 100), np.repeat([False, True], 100))
    estimate, status = adjust(.9, fit)
    assert estimate > 1
    assert status == 'incompatible_out_of_range'
    assert fit_response(np.zeros(200), np.zeros(200, bool))['status'] == 'insufficient_class_support'


def test_native_blocks_do_not_become_independent_bins():
    labels = np.repeat([False, True], 100)
    fit = fit_response(np.where(labels, .8, .1), labels, np.repeat([1, 2], 100))
    assert fit['status'] == 'insufficient_class_support'
    assert fit['blocks0'] == fit['blocks1'] == 1


def test_native_half_open_counts_and_frozen_times():
    cache = dict(position=np.array([[0, 0, 0], [.25, 25, 0]]),
                 spikes=np.array([[0, 7], [.02, 7], [.24, 7], [.25, 7]]))
    bank = native_bank(cache, np.array([[0, .25, 0, 0]]), np.array([7]))
    assert bank['counts'][:2, 0].tolist() == [1, 1]
    assert bank['counts'].sum() == 2
    np.testing.assert_allclose(bank['starts_s'], np.arange(12) * .02)
    np.testing.assert_allclose(bank['truth_cm'][:, 0], 1 + 2 * np.arange(12))


def test_large_timestamp_shared_edge_not_double_counted():
    cache = dict(position=np.array([[28579., 0, 0], [28579.25, 25, 0]]),
                 spikes=np.array([[28579.096989, 17], [28579.1, 17]]))
    bank = native_bank(cache, np.array([[28579., 28579.25, 0, 0]]), np.array([17]))
    assert bank['counts'].sum() == 2
    assert bank['counts'][4:6, 0].tolist() == [1, 1]
    np.testing.assert_array_equal(bank['ends_s'][:-1], bank['starts_s'][1:])


def test_shared_cells_counts_and_generator_reproducibility():
    rates = np.array([[2., 10., 1.], [4., 1., 2.], [3., 2., 8.]])
    grid = np.array([[0., 0.], [1., 0.], [2., 0.]])
    near = np.array([False, True, False])
    a = synthetic_bank('toy', 'test_conditional', rates, rates, grid, near, np.array([0, 3, 5]), 30)
    b = synthetic_bank('toy', 'test_conditional', rates, rates, grid, near, np.array([0, 3, 5]), 30)
    np.testing.assert_array_equal(a['counts'], b['counts'])
    np.testing.assert_array_equal(a['counts'].sum(axis=1), a['drawn_totals'])
    scores = regional_scores(a['counts'], rates, near, [np.array([0, 1]), np.array([0, 1])])
    np.testing.assert_array_equal(scores[:, 0], scores[:, 1])
    np.testing.assert_array_equal(a['labels'], np.repeat([False, True], 30))


def test_conditional_transfer_failure_remains_visible():
    fit = fit_response(np.repeat([.1, .7], 100), np.repeat([False, True], 100))
    true_pi = .2
    drift_mean = (1 - true_pi) * .6 + true_pi * .8
    corrected, _ = adjust(drift_mean, fit)
    assert abs(corrected - true_pi) > .5


def test_no_forced_common_population_answer():
    labels = np.repeat([False, True], 100)
    a = fit_response(np.repeat([.1, .7], 100), labels)
    b = fit_response(np.repeat([.2, .4], 100), labels)
    assert adjust(.3, a)[0] != pytest.approx(adjust(.3, b)[0])
