from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.benchmarks import _candidate_indices_for_model as _benchmark_candidate_indices_for_model
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.duration_dynamics import attach_duration_metadata, transition_durations_s
from hipporeplayimm.encoding import EncodingConfig, EncodingModel, LogEmissionTensor, fit_place_field_encoding
from hipporeplayimm.kd_reference import KDEncodingConfig, fit_kd_place_field_encoding
from hipporeplayimm.simulation_recovery import (
    _candidate_indices_for_model as _simulation_candidate_indices_for_model,
    simulate_replay_event,
)


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

    assert _simulation_candidate_indices_for_model(BinCenterAwareModel(), emissions, bin_centers)[0].tolist() == [0, 1]
    assert _simulation_candidate_indices_for_model(LegacyModel(), emissions, bin_centers)[0].tolist() == [1]


def test_candidate_helpers_do_not_swallow_bin_center_aware_type_errors():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    class BrokenBinCenterAwareModel:
        def candidate_indices(self, observed, centers=None):
            assert observed is emissions
            if centers is not None:
                raise TypeError("internal candidate support bug")
            return [np.array([1], dtype=int)]

    for helper in (_simulation_candidate_indices_for_model, _benchmark_candidate_indices_for_model):
        with pytest.raises(TypeError, match="internal candidate support bug"):
            helper(BrokenBinCenterAwareModel(), emissions, bin_centers)


def test_candidate_helpers_pass_keyword_only_bin_centers_when_supported():
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )
    bin_centers = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    class KeywordOnlyBinCenterModel:
        def candidate_indices(self, observed, *, centers=None):
            assert observed is emissions
            assert centers is bin_centers
            return [np.array([0], dtype=int)]

    for helper in (_simulation_candidate_indices_for_model, _benchmark_candidate_indices_for_model):
        assert helper(KeywordOnlyBinCenterModel(), emissions, bin_centers)[0].tolist() == [0]


def test_simulated_replay_rejects_nonfinite_sampling_scalars_before_poisson():
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )

    with pytest.raises(ValueError, match="dt"):
        simulate_replay_event(
            encoding,
            true_model="stationary",
            n_time=1,
            dt=float("nan"),
            rng=np.random.default_rng(1),
        )

    with pytest.raises(ValueError, match="spike_rate_scale"):
        simulate_replay_event(
            encoding,
            true_model="stationary",
            n_time=1,
            dt=0.02,
            rng=np.random.default_rng(1),
            spike_rate_scale=float("nan"),
        )


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
