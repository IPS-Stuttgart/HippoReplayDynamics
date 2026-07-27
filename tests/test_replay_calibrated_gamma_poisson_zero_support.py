from __future__ import annotations

import numpy as np

import hipporeplayimm.result_improvement_extensions as extensions


def test_direct_replay_gamma_poisson_preserves_exact_zero_rate_support() -> None:
    spike_counts = np.array([[1], [0]], dtype=int)
    rates_hz = np.array([[0.0, 5.0]], dtype=float)
    bin_durations = np.array([0.2, 0.2], dtype=float)

    actual = extensions._negative_binomial_log_emissions(
        spike_counts,
        rates_hz,
        bin_durations,
        dispersion=50.0,
    )

    assert np.isneginf(actual[0, 0])
    assert np.isfinite(actual[0, 1])
    assert np.all(np.isfinite(actual[1]))
    np.testing.assert_allclose(actual[1, 0], 0.0, rtol=0.0, atol=np.finfo(float).tiny)
