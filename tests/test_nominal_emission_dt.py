from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import clusterless, encoding, kd_reference
from hipporeplayimm.data import ReplaySession, RippleEvent, SpikeMarkData
from hipporeplayimm.state_space_sparse_momentum import _duration_adjusted_decays


def _session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenSynthetic",
        path=Path("unused"),
        position=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([1], dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=np.array([1.0], dtype=float),
            marks=np.array([[0.0]], dtype=float),
            source_file="synthetic",
            source_variable="marks",
            feature_names=("amplitude",),
            cell_ids=np.array([1], dtype=int),
        ),
    )


def _sorted_encoding() -> encoding.EncodingModel:
    return encoding.EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.array([[4.0]], dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([1], dtype=int),
        config=encoding.EncodingConfig(),
    )


def _clusterless_encoding() -> clusterless.ClusterlessMarkEncoding:
    config = clusterless.ClusterlessMarkConfig(
        mark_likelihood="diagonal-gaussian",
        mark_group_by="none",
    )
    return clusterless.ClusterlessMarkEncoding(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rate_hz=np.array([4.0], dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        effective_spike_count=np.array([1.0], dtype=float),
        mark_mean=np.array([[0.0]], dtype=float),
        mark_variance=np.array([[1.0]], dtype=float),
        mark_feature_names=("amplitude",),
        spike_mark_source="synthetic:marks",
        config=config,
        mark_likelihood="diagonal-gaussian",
    )


def test_partial_final_bin_preserves_nominal_dt_across_emission_builders() -> None:
    session = _session()
    ripple = RippleEvent(0.0, 0.025, 0.0125, 0.0, 0.0, 0.0)
    time_bin_s = 0.02
    emission_config = encoding.EmissionConfig(time_bin_s=time_bin_s)
    sorted_encoding = _sorted_encoding()

    emissions = (
        encoding.build_emissions(session, sorted_encoding, ripple, emission_config),
        kd_reference.build_kd_emissions(
            session,
            sorted_encoding,
            ripple,
            time_bin_s=time_bin_s,
        ),
        clusterless.build_clusterless_mark_emissions(
            session,
            _clusterless_encoding(),
            ripple,
            emission_config,
        ),
    )

    for tensor in emissions:
        assert tensor.dt == pytest.approx(time_bin_s)
        np.testing.assert_allclose(tensor.bin_durations, np.array([0.02, 0.005]))
        np.testing.assert_allclose(tensor.transition_durations, np.array([0.0125]))


def test_partial_final_bin_uses_nominal_dt_as_legacy_decay_reference() -> None:
    emissions = encoding.build_emissions(
        _session(),
        _sorted_encoding(),
        RippleEvent(0.0, 0.025, 0.0125, 0.0, 0.0, 0.0),
        encoding.EmissionConfig(time_bin_s=0.02),
    )
    config = SimpleNamespace(
        momentum_velocity_decay=0.81,
        momentum_velocity_decay_tau_s=0.0,
    )

    decays = _duration_adjusted_decays(
        config,
        np.asarray(emissions.transition_durations, dtype=float),
        float(emissions.dt),
    )

    assert decays[0] == pytest.approx(0.81 ** (0.0125 / 0.02))
    assert decays[0] != pytest.approx(0.81)


def test_nominal_dt_runtime_patch_is_idempotent() -> None:
    builders = (
        encoding.build_emissions,
        kd_reference.build_kd_emissions,
        clusterless.build_clusterless_mark_emissions,
    )

    hipporeplayimm.apply_runtime_patches()

    assert encoding.build_emissions is builders[0]
    assert kd_reference.build_kd_emissions is builders[1]
    assert clusterless.build_clusterless_mark_emissions is builders[2]
