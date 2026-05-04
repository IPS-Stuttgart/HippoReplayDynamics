import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.kd_reference import (
    diffusion_transition_1d,
    empirical_grid_prior,
    kd_momentum_log_evidence,
    kd_random_log_evidence,
    kd_stationary_log_evidence,
    marginalize_grid_log_evidence,
    momentum_transition_1d,
)


def test_kd_random_evidence_averages_independent_time_bin_emissions():
    log_emissions = np.log(
        np.array(
            [
                [0.2, 0.8, 1.0, 0.5],
                [1.5, 0.4, 0.3, 0.8],
                [0.7, 1.1, 0.6, 0.2],
            ]
        )
    )

    expected = np.sum(logsumexp(log_emissions, axis=1) - np.log(log_emissions.shape[1]))

    assert np.isclose(kd_random_log_evidence(log_emissions), expected)


def test_kd_stationary_evidence_integrates_one_latent_position_across_time():
    log_emissions = np.log(
        np.array(
            [
                [0.2, 0.8, 1.0, 0.5],
                [1.5, 0.4, 0.3, 0.8],
                [0.7, 1.1, 0.6, 0.2],
            ]
        )
    )

    expected = logsumexp(np.sum(log_emissions, axis=0) - np.log(log_emissions.shape[1]))

    assert np.isclose(kd_stationary_log_evidence(log_emissions), expected)


def test_diffusion_transition_columns_normalize():
    transition = diffusion_transition_1d(n_bins=5, sd_meters=0.3, bin_size_cm=4.0, dt=0.003)

    assert np.allclose(transition.sum(axis=0), 1.0)
    assert np.all(transition >= 0.0)


def test_momentum_transition_normalizes_and_changes_with_decay():
    slow_decay = momentum_transition_1d(n_bins=5, sd_meters=40.0, decay=1.0, bin_size_cm=4.0, dt=0.003)
    fast_decay = momentum_transition_1d(n_bins=5, sd_meters=40.0, decay=100.0, bin_size_cm=4.0, dt=0.003)

    assert np.allclose(slow_decay.sum(axis=0), 1.0)
    assert np.allclose(fast_decay.sum(axis=0), 1.0)
    assert not np.allclose(slow_decay, fast_decay)


def test_momentum_log_evidence_is_finite_on_tiny_grid():
    log_emissions = np.log(
        np.array(
            [
                [0.2, 0.8, 1.0, 0.5],
                [1.5, 0.4, 0.3, 0.8],
                [0.7, 1.1, 0.6, 0.2],
                [0.3, 0.9, 1.2, 0.4],
            ]
        )
    )

    value = kd_momentum_log_evidence(
        log_emissions,
        n_bins_x=2,
        n_bins_y=2,
        sd_meters=40.0,
        decay=25.0,
        initial_sd_m_per_s=10.0,
        bin_size_cm=4.0,
        dt=0.003,
    )

    assert np.isfinite(value)


def test_grid_marginalization_returns_finite_log_evidence_per_event():
    grid = np.array(
        [
            [-10.0, -9.0, -11.0],
            [-8.0, -7.5, -8.5],
            [-12.0, -11.0, -10.5],
        ]
    )
    prior, _ = empirical_grid_prior({"sd_meters": np.array([0.1, 0.2, 0.3])}, grid)

    marginalized = marginalize_grid_log_evidence(grid, prior)

    assert marginalized.shape == (grid.shape[0],)
    assert np.all(np.isfinite(marginalized))
