from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.accuracy_upgrades import (
    masked_gaussian_transition_matrix,
    negative_binomial_log_emissions,
)


def test_masked_gaussian_transition_matrix_rejects_invalid_bin_centers() -> None:
    with pytest.raises(ValueError, match="shape"):
        masked_gaussian_transition_matrix(np.empty((2, 0)), sigma_cm=1.0)

    centers_with_nan = np.asarray([[0.0, 0.0], [np.nan, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        masked_gaussian_transition_matrix(centers_with_nan, sigma_cm=1.0)


def test_masked_gaussian_transition_matrix_rejects_nonfinite_or_nonpositive_scales() -> None:
    centers = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    for sigma_cm in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="sigma_cm"):
            masked_gaussian_transition_matrix(centers, sigma_cm=sigma_cm)

    for max_step_sigma in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="max_step_sigma"):
            masked_gaussian_transition_matrix(
                centers,
                sigma_cm=1.0,
                max_step_sigma=max_step_sigma,
            )


def test_negative_binomial_log_emissions_rejects_invalid_scale_parameters() -> None:
    spike_counts = np.zeros((2, 1), dtype=float)
    rates_hz = np.ones((1, 3), dtype=float)

    for overdispersion in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="overdispersion"):
            negative_binomial_log_emissions(
                spike_counts,
                rates_hz,
                0.02,
                overdispersion=overdispersion,
            )

    for spike_rate_scale in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="spike_rate_scale"):
            negative_binomial_log_emissions(
                spike_counts,
                rates_hz,
                0.02,
                overdispersion=0.5,
                spike_rate_scale=spike_rate_scale,
            )
