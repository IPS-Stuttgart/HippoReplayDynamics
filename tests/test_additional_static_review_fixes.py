from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.duration_dynamics import attach_duration_metadata, transition_durations_s
from hipporeplayimm.encoding import EncodingConfig, LogEmissionTensor, fit_place_field_encoding
from hipporeplayimm.kd_reference import KDEncodingConfig, fit_kd_place_field_encoding
from hipporeplayimm.simulation_recovery import _candidate_indices_for_model


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _empty_spike_session() -> ReplaySession:
    times = np.linspace(0.0, 2.0, 21)
    position = np.column_stack(
        [
            times,
            np.linspace(0.0, 20.0, times.shape[0]),
            np.linspace(0.0, 5.0, times.shape[0]),
        ]
    )
    return ReplaySession(
        rat="RatX",
        name="Open1",
        path=Path("RatX/Open1"),
        position=position,
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def test_place_field_encoding_smoothing_handles_empty_cell_set():
    session = _empty_spike_session()

    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(bin_size_cm=5.0, smoothing_sigma_bins=1.0, min_speed_cm_s=0.0),
    )

    assert encoding.n_cells == 0
    assert encoding.rates_hz.shape == (0, encoding.n_bins)


def test_kd_place_field_encoding_smoothing_handles_empty_cell_set():
    session = _empty_spike_session()

    encoding = fit_kd_place_field_encoding(
        session,
        KDEncodingConfig(
            bin_size_cm=5.0,
            n_bins_x=8,
            n_bins_y=8,
            smoothing_sigma_cm=5.0,
            min_speed_cm_s=0.0,
        ),
    )

    assert encoding.n_cells == 0
    assert encoding.rates_hz.shape == (0, encoding.n_bins)


def test_simulation_recovery_candidate_helper_passes_bin_centers_when_supported():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    class BinCenterAwareModel:
        def candidate_indices(self, observed, centers=None):
            assert observed is emissions
            assert centers is bin_centers
            return [np.array([0, 1], dtype=int)]

    class LegacyModel:
        def candidate_indices(self, observed):
            assert observed is emissions
            return [np.array([1], dtype=int)]

    assert _candidate_indices_for_model(BinCenterAwareModel(), emissions, bin_centers)[0].tolist() == [0, 1]
    assert _candidate_indices_for_model(LegacyModel(), emissions, bin_centers)[0].tolist() == [1]


def test_track_event_prefix_emissions_slices_duration_metadata():
    track_event = _load_script_module("track_event_under_test", ROOT / "scripts" / "track_event.py")
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((5, 3), dtype=float),
        spike_counts=np.zeros((5, 1), dtype=int),
        times=np.array([0.05, 0.15, 0.25, 0.40, 0.55], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    emissions.transition_durations = np.array([0.1, 0.1, 0.15, 0.15], dtype=float)
    attach_duration_metadata(emissions)

    prefix = track_event._prefix_emissions(emissions, 3)

    np.testing.assert_allclose(transition_durations_s(prefix), np.array([0.1, 0.1]))
    assert tuple(prefix.dt.transition_durations) == (0.1, 0.1)
    assert prefix.n_time == 3
    assert prefix.n_spikes == 0
