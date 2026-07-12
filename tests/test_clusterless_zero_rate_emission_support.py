from __future__ import annotations

from pathlib import Path

import numpy as np

from hipporeplayimm.clusterless import (
    ClusterlessMarkConfig,
    ClusterlessMarkEncoding,
    build_clusterless_mark_emissions,
)
from hipporeplayimm.data import ReplaySession, RippleEvent, SpikeMarkData
from hipporeplayimm.encoding import EmissionConfig


def _session(mark_time: float, group_id: int | None = None) -> ReplaySession:
    group_ids = None if group_id is None else np.array([group_id], dtype=int)
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("synthetic"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=np.array([mark_time], dtype=float),
            marks=np.array([[0.0]], dtype=float),
            source_file="synthetic.mat",
            source_variable="Marks",
            feature_names=("mark0",),
            cell_ids=np.array([1], dtype=int),
            group_ids=group_ids,
        ),
    )


def _encoding(
    rate_hz: list[float],
    *,
    group_rate_hz: list[list[float]] | None = None,
) -> ClusterlessMarkEncoding:
    grouped = group_rate_hz is not None
    return ClusterlessMarkEncoding(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]]),
        rate_hz=np.asarray(rate_hz, dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        effective_spike_count=np.ones(2, dtype=float),
        mark_mean=np.zeros((2, 1), dtype=float),
        mark_variance=np.ones((2, 1), dtype=float),
        mark_feature_names=("mark0",),
        spike_mark_source="synthetic:Marks",
        config=ClusterlessMarkConfig(
            mark_likelihood="diagonal-gaussian",
            mark_group_by="tetrode" if grouped else "none",
        ),
        mark_likelihood="diagonal-gaussian",
        group_ids=np.array([7], dtype=int) if grouped else None,
        group_rate_hz=(
            np.asarray(group_rate_hz, dtype=float)
            if group_rate_hz is not None
            else None
        ),
        group_effective_spike_count=(
            np.ones((1, 2), dtype=float) if grouped else None
        ),
        group_mark_mean=(np.zeros((1, 2, 1), dtype=float) if grouped else None),
        group_mark_variance=(np.ones((1, 2, 1), dtype=float) if grouped else None),
    )


def _ripple() -> RippleEvent:
    return RippleEvent(
        start=0.0,
        end=1.0,
        peak=0.5,
        raw_power=0.0,
        z_power_session=0.0,
        z_power_epoch=0.0,
    )


def test_clusterless_spike_at_zero_global_rate_is_impossible() -> None:
    emissions = build_clusterless_mark_emissions(
        _session(mark_time=0.5),
        _encoding([0.0, 1.0]),
        _ripple(),
        EmissionConfig(time_bin_s=1.0),
    )

    assert np.isneginf(emissions.log_likelihood[0, 0])
    assert np.isfinite(emissions.log_likelihood[0, 1])


def test_clusterless_silence_at_zero_global_rate_has_unit_probability() -> None:
    emissions = build_clusterless_mark_emissions(
        _session(mark_time=2.0),
        _encoding([0.0, 1.0]),
        _ripple(),
        EmissionConfig(time_bin_s=1.0),
    )

    assert emissions.log_likelihood[0, 0] == 0.0
    np.testing.assert_allclose(emissions.log_likelihood[0, 1], -1.0)


def test_clusterless_spike_at_zero_group_rate_is_impossible() -> None:
    emissions = build_clusterless_mark_emissions(
        _session(mark_time=0.5, group_id=7),
        _encoding([1.0, 1.0], group_rate_hz=[[0.0, 1.0]]),
        _ripple(),
        EmissionConfig(time_bin_s=1.0),
    )

    assert np.isneginf(emissions.log_likelihood[0, 0])
    assert np.isfinite(emissions.log_likelihood[0, 1])
