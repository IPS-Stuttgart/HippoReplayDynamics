import numpy as np
import pytest

from scripts.audit_run_error_content_diagnostic import check_decoding, gradients, predictions, spatial_tiles, verify_fit
from scripts.measure_population_content_stability import readouts, tile_ids
from scripts.run_error_content_diagnostic import MODELS, code_gradient, fit, predict
from tests.test_run_error_content_diagnostic import training


def test_normal_equation_independently_recovers_all_models():
    frame = training()
    for name in MODELS:
        state = fit(frame, name)
        verify_fit(frame, state)
        np.testing.assert_allclose(predictions(frame, state), predict(frame, state), atol=1e-12)


def test_direct_neighbor_gradient_matches_edge_accumulation():
    grid = np.array([[0., 0.], [8., 0.], [0., 8.], [8., 8.], [32., 8.]])
    rates = np.random.default_rng(81).gamma(2, 3, (11, len(grid)))
    np.testing.assert_allclose(gradients(rates, grid), code_gradient(rates, grid), atol=1e-14)


def test_independent_posterior_and_truth_tile_audit_detects_corruption():
    grid = np.array([[0., 0.], [8., 0.], [0., 8.], [8., 8.]])
    rates = np.array([[1., 8., 1., 2.], [9., 1., 3., 1.]])
    ca, cb = np.array([[1, 0], [0, 3]]), np.array([[2, 1], [0, 2]])
    truth = grid[[0, 3]]
    frame = readouts(ca, cb, rates, rates, grid, grid, np.arange(4.), truth)
    frame['event_index'] = [12, 34]
    frame['true_tile'] = tile_ids(grid)[[0, 3]]
    check_decoding(frame, ca, cb, rates, rates, grid, truth, np.array([12, 34]))
    for key in ('b_truth_error_cm', 'regional_tv', 'a_entropy', 'true_tile'):
        broken = frame.copy()
        broken.loc[0, key] += 1
        with pytest.raises(AssertionError):
            check_decoding(broken, ca, cb, rates, rates, grid, truth, np.array([12, 34]))


def test_tile_edges_follow_frozen_numerical_order():
    x, y = np.meshgrid(np.arange(0, 209, 8), np.arange(0, 161, 8))
    grid = np.c_[x.ravel(), y.ravel()]
    np.testing.assert_array_equal(spatial_tiles(grid), tile_ids(grid))
