from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.data import ReplaySession, RippleEvent
from hipporeplayimm.encoding import EncodingModel
from hipporeplayimm.kd_reference import (
    KDEncodingConfig,
    build_kd_emissions,
    diffusion_transition_1d,
    empirical_grid_prior,
    kd_momentum_log_evidence,
    kd_random_log_evidence,
    kd_stationary_gaussian_log_evidence_from_latent,
    kd_stationary_gaussian_log_evidence_from_transitions,
    kd_stationary_log_evidence,
    marginalize_grid_log_evidence,
    momentum_transition_1d,
    poisson_log_emissions,
    stationary_gaussian_log_latent,
    stationary_gaussian_transition_1d,
)


def _one_cell_encoding(rate_hz: float = 10.0) -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 4.0]),
        y_edges=np.array([0.0, 4.0]),
        bin_centers=np.array([[2.0, 2.0]]),
        rates_hz=np.array([[rate_hz]]),
        occupancy_s=np.array([1.0]),
        cell_ids=np.array([1]),
        config=KDEncodingConfig(),
    )


def _session_with_ripple(spikes: np.ndarray, ripple: RippleEvent) -> ReplaySession:
    return ReplaySession(
        rat="rat",
        name="session",
        path=Path("."),
        position=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        spikes=np.asarray(spikes, dtype=float),
        tetrode_cell_ids=np.empty((0, 2)),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.array(
            [[ripple.start, ripple.end, ripple.peak, ripple.raw_power, ripple.z_power_session, ripple.z_power_epoch]]
        ),
        run_times=np.empty((0, 2)),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
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


def test_stationary_gaussian_separable_evidence_matches_dense_latent():
    log_emissions = np.log(
        np.array(
            [
                [0.2, 0.8, 1.0, 0.5],
                [1.5, 0.4, 0.3, 0.8],
                [0.7, 1.1, 0.6, 0.2],
            ]
        )
    )
    latent = stationary_gaussian_log_latent(n_bins_x=2, n_bins_y=2, sd_meters=0.1, bin_size_cm=4.0)
    transition = stationary_gaussian_transition_1d(n_bins=2, sd_meters=0.1, bin_size_cm=4.0)

    dense = kd_stationary_gaussian_log_evidence_from_latent(log_emissions, latent)
    separable = kd_stationary_gaussian_log_evidence_from_transitions(log_emissions, 2, 2, transition)

    assert np.isclose(separable, dense)


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


def test_build_kd_emissions_keeps_last_full_bin_when_duration_is_multiple():
    ripple = RippleEvent(start=0.0, end=0.1, peak=0.05, raw_power=0.0, z_power_session=0.0, z_power_epoch=0.0)
    session = _session_with_ripple(np.array([[0.09, 1.0], [0.101, 1.0]]), ripple)

    emissions = build_kd_emissions(session, _one_cell_encoding(), ripple, time_bin_s=0.02)

    assert emissions.n_time == 5
    assert np.allclose(emissions.times, np.array([0.01, 0.03, 0.05, 0.07, 0.09]))
    assert np.array_equal(emissions.spike_counts[:, 0], np.array([0, 0, 0, 0, 1]))
    assert emissions.n_spikes == 1


def test_build_kd_emissions_includes_partial_ripple_tail_without_counting_after_end():
    ripple = RippleEvent(start=0.0, end=0.053, peak=0.026, raw_power=0.0, z_power_session=0.0, z_power_epoch=0.0)
    session = _session_with_ripple(np.array([[0.051, 1.0], [0.054, 1.0]]), ripple)
    encoding = _one_cell_encoding(rate_hz=10.0)

    emissions = build_kd_emissions(session, encoding, ripple, time_bin_s=0.02)

    assert emissions.n_time == 3
    assert np.allclose(emissions.times, np.array([0.01, 0.03, 0.0465]))
    assert np.array_equal(emissions.spike_counts[:, 0], np.array([0, 0, 1]))
    assert emissions.n_spikes == 1

    expected_log_likelihood = poisson_log_emissions(
        np.array([[0], [0], [1]]),
        encoding.rates_hz,
        np.array([0.02, 0.02, 0.013]),
    )
    assert np.allclose(emissions.log_likelihood, expected_log_likelihood)
