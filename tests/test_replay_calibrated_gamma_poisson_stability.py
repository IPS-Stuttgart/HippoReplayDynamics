from __future__ import annotations

import numpy as np
from scipy.special import gammaln

import hipporeplayimm
import hipporeplayimm.result_improvement_extensions as extensions


def _poisson_log_emissions(
    spike_counts: np.ndarray,
    rates_hz: np.ndarray,
    bin_durations: np.ndarray,
) -> np.ndarray:
    mean = bin_durations[:, None, None] * rates_hz[None, :, :]
    counts = spike_counts[:, :, None]
    return np.sum(
        counts * np.log(mean) - mean - gammaln(counts + 1.0),
        axis=1,
    )


def test_replay_gamma_poisson_large_dispersion_reaches_poisson_limit() -> None:
    spike_counts = np.array([[0, 1], [2, 3]], dtype=int)
    rates_hz = np.array(
        [
            [2.0, 5.0, 9.0],
            [7.0, 11.0, 13.0],
        ],
        dtype=float,
    )
    bin_durations = np.array([0.25, 0.5], dtype=float)

    actual = extensions._negative_binomial_log_emissions(
        spike_counts,
        rates_hz,
        bin_durations,
        dispersion=1.0e300,
    )
    expected = _poisson_log_emissions(spike_counts, rates_hz, bin_durations)

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=5.0e-12)


def test_replay_gamma_poisson_stability_patch_is_idempotent() -> None:
    patched = extensions._negative_binomial_log_emissions

    hipporeplayimm.apply_runtime_patches()

    assert extensions._negative_binomial_log_emissions is patched
