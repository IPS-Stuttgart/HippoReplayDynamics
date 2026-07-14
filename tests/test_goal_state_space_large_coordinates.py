import numpy as np

from hipporeplayimm.goal_state_space import _goal_drift_prediction, _goal_transition_matrix


def test_goal_drift_prediction_preserves_large_finite_step():
    step = 1e199

    predicted = _goal_drift_prediction(
        np.array([0.0, 0.0]),
        np.array([1e200, 1e200]),
        step,
    )

    assert np.all(np.isfinite(predicted))
    assert np.allclose(predicted, np.full(2, step / np.sqrt(2.0)), rtol=1e-14)


def test_goal_drift_prediction_handles_opposite_sign_extremes():
    predicted = _goal_drift_prediction(
        np.array([1e308, 0.0]),
        np.array([-1e308, 0.0]),
        1e307,
    )

    assert np.all(np.isfinite(predicted))
    assert np.allclose(predicted, np.array([9e307, 0.0]), rtol=1e-14)


def test_goal_transition_matrix_preserves_large_scale_gaussian_weights():
    centers = np.array(
        [
            [0.0, 0.0],
            [1e200, 0.0],
            [2e200, 0.0],
        ]
    )

    transition = _goal_transition_matrix(
        centers,
        np.array([0.0, 0.0]),
        drift_step_cm=0.0,
        sigma_cm=1e200,
        max_step_sigma=4.0,
    ).toarray()

    expected = np.exp(-0.5 * np.array([0.0, 1.0, 4.0]))
    expected /= expected.sum()
    assert np.all(np.isfinite(transition))
    assert np.all(transition >= 0.0)
    assert np.allclose(transition.sum(axis=0), 1.0)
    assert np.allclose(transition[:, 0], expected, rtol=1e-14)
