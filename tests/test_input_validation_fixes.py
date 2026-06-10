from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession, RippleEvent, _as_intervals, _as_two_dimensional
from hipporeplayimm.encoding import (
    EncodingConfig,
    EncodingModel,
    LogEmissionTensor,
    _poisson_log_emissions,
    fit_place_field_encoding,
)
from hipporeplayimm.kd_reference import build_kd_emissions, poisson_log_emissions as _kd_poisson_log_emissions
from hipporeplayimm.models import CandidateKinematicModel
from hipporeplayimm.state_space_utils import _full_grid_normalized_pairwise_gaussian_log_prob, _gaussian_transition_matrix


def _minimal_session(position: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenY",
        path=Path("unused"),
        position=np.asarray(position, dtype=float),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 1.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def test_fit_place_field_encoding_rejects_malformed_position_array() -> None:
    session = _minimal_session(np.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="position"):
        fit_place_field_encoding(session)


def test_fit_place_field_encoding_rejects_nonpositive_bin_size() -> None:
    session = _minimal_session(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        )
    )

    with pytest.raises(ValueError, match="bin_size_cm"):
        fit_place_field_encoding(session, EncodingConfig(bin_size_cm=0.0))


def test_encoding_model_positions_to_flat_bins_rejects_bad_shape() -> None:
    model = EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.empty((0, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([], dtype=int),
        config=EncodingConfig(),
    )

    with pytest.raises(ValueError, match="xy"):
        model.positions_to_flat_bins(np.array([0.5, 0.5]))


def test_empty_ripple_events_keep_six_column_schema() -> None:
    ripple_events = _as_two_dimensional(np.array([]), "Ripple_Events")

    assert ripple_events.shape == (0, 6)


def test_ripple_events_column_major_schema_is_transposed() -> None:
    ripple_events = _as_two_dimensional(np.arange(12.0).reshape(6, 2), "Ripple_Events")

    assert ripple_events.shape == (2, 6)
    np.testing.assert_allclose(ripple_events[0], np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]))


def test_malformed_ripple_events_are_rejected() -> None:
    with pytest.raises(ValueError, match="Ripple_Events"):
        _as_two_dimensional(np.zeros((2, 5)), "Ripple_Events")
    with pytest.raises(ValueError, match="Ripple_Events"):
        _as_two_dimensional(np.zeros(2), "Ripple_Events")


def test_interval_arrays_reject_higher_dimensional_inputs() -> None:
    with pytest.raises(ValueError, match="Intervals"):
        _as_intervals(np.zeros((1, 2, 2)))


def test_candidate_kinematic_model_rejects_negative_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        CandidateKinematicModel(mode="imm", top_k=-1)


def test_state_space_gaussian_transition_rejects_nonfinite_parameters() -> None:
    bin_centers = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="sigma_cm"):
        _gaussian_transition_matrix(bin_centers, float("nan"), 4.0)

    with pytest.raises(ValueError, match="max_step_sigma"):
        _gaussian_transition_matrix(bin_centers, 1.0, float("nan"))


def test_log_emission_tensor_rejects_nonfinite_duration_metadata() -> None:
    log_likelihood = np.zeros((2, 1), dtype=float)
    spike_counts = np.zeros((2, 1), dtype=int)
    times = np.array([0.0, 0.02], dtype=float)
    cell_ids = np.array([1], dtype=int)

    with pytest.raises(ValueError, match="dt"):
        LogEmissionTensor(
            log_likelihood=log_likelihood,
            spike_counts=spike_counts,
            times=times,
            dt=float("nan"),
            cell_ids=cell_ids,
            n_spikes=0,
        )

    with pytest.raises(ValueError, match="bin_durations"):
        LogEmissionTensor(
            log_likelihood=log_likelihood,
            spike_counts=spike_counts,
            times=times,
            dt=0.02,
            cell_ids=cell_ids,
            n_spikes=0,
            bin_durations=np.array([0.02, float("inf")], dtype=float),
        )

    with pytest.raises(ValueError, match="transition_durations"):
        LogEmissionTensor(
            log_likelihood=log_likelihood,
            spike_counts=spike_counts,
            times=times,
            dt=0.02,
            cell_ids=cell_ids,
            n_spikes=0,
            transition_durations=np.array([0.0], dtype=float),
        )


def test_sorted_poisson_log_emissions_reject_nonfinite_durations() -> None:
    spike_counts = np.zeros((2, 1), dtype=int)
    rates_hz = np.ones((1, 2), dtype=float)

    with pytest.raises(ValueError, match="finite and positive"):
        _poisson_log_emissions(spike_counts, rates_hz, float("nan"))

    with pytest.raises(ValueError, match="finite and positive"):
        _poisson_log_emissions(spike_counts, rates_hz, np.array([0.02, float("inf")], dtype=float))


def test_kd_poisson_log_emissions_reject_nonfinite_inputs() -> None:
    spike_counts = np.zeros((2, 1), dtype=int)
    rates_hz = np.ones((1, 2), dtype=float)

    with pytest.raises(ValueError, match="spike_rate_scale"):
        _kd_poisson_log_emissions(spike_counts, rates_hz, 0.02, spike_rate_scale=float("nan"))

    with pytest.raises(ValueError, match="finite and positive"):
        _kd_poisson_log_emissions(spike_counts, rates_hz, float("nan"))

    with pytest.raises(ValueError, match="finite and positive"):
        _kd_poisson_log_emissions(spike_counts, rates_hz, np.array([0.02, float("inf")], dtype=float))


def test_kd_emissions_preserve_partial_bin_duration_metadata() -> None:
    session = _minimal_session(np.array([[0.0, 0.0, 0.0], [0.05, 1.0, 1.0]], dtype=float))
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )
    ripple = RippleEvent(start=0.0, end=0.05, peak=0.025, raw_power=0.0, z_power_session=0.0, z_power_epoch=0.0)

    emissions = build_kd_emissions(session, encoding, ripple, time_bin_s=0.02)

    np.testing.assert_allclose(emissions.bin_durations, np.array([0.02, 0.02, 0.01]))
    np.testing.assert_allclose(emissions.transition_durations, np.diff(emissions.times))


def test_pairwise_gaussian_log_prob_rejects_nonpositive_sigma() -> None:
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="sigma_cm"):
        _full_grid_normalized_pairwise_gaussian_log_prob(bin_centers[:1], bin_centers, bin_centers, 0.0)
