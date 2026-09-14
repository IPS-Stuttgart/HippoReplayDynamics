import numpy as np
import pandas as pd
import pytest

from scripts.run_error_content_diagnostic import BASE, FULL, code_gradient, fit, predict, predictor_features


def training():
    n = 120
    rng = np.random.default_rng(81)
    f = pd.DataFrame({name: rng.random(n) for name in FULL})
    f['animal'] = np.repeat(['a', 'b', 'c'], 40)
    f['session'] = f.animal
    f['dataset'], f['source'], f['split'], f['draw'] = 'pfeiffer_foster', 'run_calibration', 0, -1
    f['event_index'] = np.arange(n)
    f['grid_diagonal_cm'] = 200.
    f['b_truth_error_cm'] = 50. * f.a_entropy
    return f


def test_replay_or_external_labels_cannot_train():
    frame = training()
    for key, value in [('source', 'real'), ('dataset', 'tanni2022'), ('split', 1)]:
        altered = frame.copy(); altered[key] = value
        with pytest.raises(ValueError): fit(altered, 'full')


def test_predictions_do_not_read_B_replay_or_truth():
    frame = training()
    state = fit(frame, 'full')
    expected = predict(frame, state)
    frame['b_truth_error_cm'] = 1e10
    frame['b_entropy'], frame['endpoint_separation_cm'], frame['b_spikes'] = -999., 99999., 999999
    np.testing.assert_array_equal(predict(frame, state), expected)


def test_feature_construction_ignores_B_replay():
    grid = np.array([[0., 0.], [8., 0.], [16., 0.], [16., 8.]])
    rates_a = np.array([[1., 5., 1., 2.], [2., 1., 4., 1.]])
    frame = pd.DataFrame(dict(a_spikes=[3, 5], a_active=[2, 1], a_entropy=[.2, .7], a_width_cm=[4., 8.],
                             a_peak=[.6, .4], a_x_cm=[8., 13.], a_y_cm=[0., 2.]))
    before = predictor_features(frame, rates_a, rates_a[::-1], grid)
    frame['b_spikes'], frame['b_x_cm'], frame['b_entropy'] = 999, -1000., -1000.
    pd.testing.assert_frame_equal(before, predictor_features(frame, rates_a, rates_a[::-1], grid))
    assert before.columns.tolist() == list(FULL)
    assert not any(name.startswith('b_') for name in BASE)


def test_code_gradient_known_geometry():
    gradient = code_gradient(np.array([[1., 9., 25.]]), np.array([[0., 0.], [8., 0.], [16., 0.]]))
    np.testing.assert_allclose(gradient, 4 / 64)


def test_frozen_preprocessing_and_mean_baseline():
    frame = training()
    full = fit(frame, 'full')
    original = predict(frame, full)
    altered = frame.copy(); altered['a_entropy'] = 1e5
    predict(altered, full)
    np.testing.assert_array_equal(predict(frame, full), original)
    mean = fit(frame, 'mean')
    assert np.ptp(predict(frame, mean)) == 0
