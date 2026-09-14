import numpy as np
import pandas as pd
import pytest

from scripts.audit_local_run_content_risk import dense_calibration, matrix, verify_fit
from scripts.local_run_content_risk import FULL, calibration, features, fit
from scripts.verify_kleinman_content_readiness import direct_counts, direct_immobile


def test_direct_spike_verifier_boundaries_and_compound_identity():
    spikes = np.array([[0.1, 1, 1], [0.11, 1, 2], [0.12, 1, 1], [0.105, 2, 1]])
    assert direct_counts(spikes, {(1, 1), (1, 2)}, 0.1, 0.12) == (2, 2)
    assert direct_counts(spikes, {(2, 1)}, 0.1, 0.12) == (1, 1)
    v = np.array([[0.0, 0.0], [0.02, 0.0], [0.04, 6.0], [0.24, 0.0]])
    assert direct_immobile(v, 0.001, 0.019)
    assert not direct_immobile(v, 0.021, 0.039)
    assert not direct_immobile(v, 0.05, 0.1)
    assert not direct_immobile(v, -0.01, 0.01)


def test_independent_dense_calibration_and_closed_form_ridge():
    rng = np.random.default_rng(8)
    grid = np.array([[x, y] for x in range(0, 32, 8) for y in range(0, 32, 8)], float)
    rates = rng.uniform(0.2, 15, (8, len(grid)))
    counts = rng.poisson(0.2, (120, 8))
    truth = rng.uniform(0, 24, (120, 2))
    windows = np.column_stack([np.arange(120), np.arange(120) + 0.02, truth])
    arrays = {"calibration_counts": counts, "calibration_windows": windows, "rates_hz": rates, "grid_cm": grid}
    part = {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7]}
    frame, y = dense_calibration(arrays, part["a"], "a")
    x, error = calibration(arrays, part, "a")
    expected = np.log1p(error / np.linalg.norm(np.ptp(grid, axis=0)))
    np.testing.assert_allclose(y, expected, atol=1e-12, rtol=1e-12)
    actual_x = matrix(frame, "a", rates[part["a"]], grid, FULL)
    np.testing.assert_allclose(x[list(FULL)], actual_x, atol=1e-12, rtol=1e-12)
    pd.testing.assert_frame_equal(x, features(frame, "a", rates[part["a"]], grid), atol=1e-12, rtol=1e-12)
    state = fit(x, error, np.linalg.norm(np.ptp(grid, axis=0)), "full")
    verify_fit(actual_x, y, state)
    state["coefficients"][0] += 0.01
    with pytest.raises(AssertionError):
        verify_fit(actual_x, y, state)
