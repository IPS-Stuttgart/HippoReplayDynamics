from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.accuracy_upgrades import (
    ReplayGainConfig,
    build_continuous_time_emissions,
    estimate_replay_cell_gains,
    gamma_poisson_predictive_log_emissions,
    masked_gaussian_transition_matrix,
    negative_binomial_log_emissions,
)
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def _continuous_time_session(spike_cell_id: float) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenTest",
        path=Path("."),
        position=np.empty((0, 3), dtype=float),
        spikes=np.array([[0.10, spike_cell_id], [0.20, 2.0]], dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.array([[0.0, 0.30, 0.15, 0.0, 0.0, 0.0]], dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _two_cell_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((2, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1, 2], dtype=int),
        config=EncodingConfig(),
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


@pytest.mark.parametrize("bad_cell_id", [1.5, 1.0000000005])
def test_continuous_time_emissions_reject_nonintegral_spike_cell_ids(bad_cell_id: float) -> None:
    with pytest.raises(ValueError, match="spike cell IDs"):
        build_continuous_time_emissions(_continuous_time_session(bad_cell_id), _two_cell_encoding(), 0)


def test_estimate_replay_cell_gains_counts_unsorted_encoding_cell_ids() -> None:
    session = ReplaySession(
        rat="RatX",
        name="OpenTest",
        path=Path("."),
        position=np.empty((0, 3), dtype=float),
        spikes=np.array([[0.10, 20], [0.20, 10], [0.30, 20], [0.40, 99]], dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.array([[0.0, 0.35, 0.2, 0.0, 0.0, 0.0]], dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.zeros((2, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([20, 10], dtype=int),
        config=EncodingConfig(),
    )

    gains = estimate_replay_cell_gains(
        session,
        encoding,
        [0],
        ReplayGainConfig(
            prior_observed_spikes=0.0,
            prior_expected_spikes=1.0,
            min_gain=0.0,
            max_gain=10.0,
        ),
    )

    np.testing.assert_allclose(gains, [2.0, 1.0])


def test_gamma_poisson_predictive_log_emissions_rejects_boolean_inputs() -> None:
    spike_counts = np.zeros((2, 1), dtype=float)
    shape = np.ones((1, 3), dtype=float)
    exposure = np.ones((1, 3), dtype=float)

    with pytest.raises(ValueError, match="spike_counts"):
        gamma_poisson_predictive_log_emissions(
            np.array([[True], [False]], dtype=bool),
            shape,
            exposure,
            0.02,
        )

    with pytest.raises(ValueError, match="rate_shape"):
        gamma_poisson_predictive_log_emissions(
            spike_counts,
            np.array([[True, True, True]], dtype=bool),
            exposure,
            0.02,
        )

    with pytest.raises(ValueError, match="rate_exposure_s"):
        gamma_poisson_predictive_log_emissions(
            spike_counts,
            shape,
            np.array([[True, True, True]], dtype=bool),
            0.02,
        )

    with pytest.raises(ValueError, match="dt"):
        gamma_poisson_predictive_log_emissions(spike_counts, shape, exposure, True)

    with pytest.raises(ValueError, match="spike_rate_scale"):
        gamma_poisson_predictive_log_emissions(
            spike_counts,
            shape,
            exposure,
            0.02,
            spike_rate_scale=True,
        )


def test_gamma_poisson_predictive_log_emissions_rejects_invalid_inputs() -> None:
    spike_counts = np.zeros((2, 1), dtype=float)
    shape = np.ones((1, 3), dtype=float)
    exposure = np.ones((1, 3), dtype=float)

    for spike_rate_scale in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="spike_rate_scale"):
            gamma_poisson_predictive_log_emissions(
                spike_counts,
                shape,
                exposure,
                0.02,
                spike_rate_scale=spike_rate_scale,
            )

    for dt in (0.0, -0.01, np.nan, np.inf):
        with pytest.raises(ValueError, match="dt"):
            gamma_poisson_predictive_log_emissions(spike_counts, shape, exposure, dt)

    with pytest.raises(ValueError, match="rate_shape"):
        gamma_poisson_predictive_log_emissions(
            spike_counts,
            np.array([[0.0, 1.0, 1.0]], dtype=float),
            exposure,
            0.02,
        )

    with pytest.raises(ValueError, match="rate_exposure_s"):
        gamma_poisson_predictive_log_emissions(
            spike_counts,
            shape,
            np.array([[1.0, -1.0, 1.0]], dtype=float),
            0.02,
        )
