from __future__ import annotations

from pathlib import Path

import numpy as np

import hipporeplayimm
from hipporeplayimm import accuracy_upgrades
from hipporeplayimm.accuracy_upgrades import ContinuousTimeEmissionConfig
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EncodingModel


def _session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenTest",
        path=Path("."),
        position=np.empty((0, 3), dtype=float),
        spikes=np.array([[0.1000, 1.0], [0.1005, 1.0]], dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.array([[0.0, 0.2, 0.1, 0.0, 0.0, 0.0]], dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


def _encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=EncodingConfig(),
    )


def test_continuous_time_dynamics_use_recorded_timestamp_differences() -> None:
    emissions = accuracy_upgrades.build_continuous_time_emissions(
        _session(),
        _encoding(),
        0,
        ContinuousTimeEmissionConfig(min_interval_s=0.01),
    )

    # The observation bin remains clamped for numerical stability, but dynamics
    # must evolve over the actual 0.5 ms between adjacent spike timestamps.
    assert emissions.bin_durations[1] == 0.01
    np.testing.assert_allclose(
        emissions.transition_durations,
        np.diff(emissions.times),
        rtol=0.0,
        atol=1e-15,
    )
    assert emissions.transition_durations[0] < 0.01


def test_continuous_time_duration_patch_is_idempotent() -> None:
    active = accuracy_upgrades.build_continuous_time_emissions

    hipporeplayimm.apply_runtime_patches()

    assert accuracy_upgrades.build_continuous_time_emissions is active
