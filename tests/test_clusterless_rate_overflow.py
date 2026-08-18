from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.clusterless import (
    ClusterlessMarkConfig,
    ClusterlessMarkEncoding,
    build_clusterless_mark_emissions,
)
from hipporeplayimm.data import ReplaySession, RippleEvent, SpikeMarkData
from hipporeplayimm.encoding import EmissionConfig


def _encoding(rate_hz: float, *, group_rate_hz: float = 1.0) -> ClusterlessMarkEncoding:
    return ClusterlessMarkEncoding(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.0, 0.0]]),
        rate_hz=np.array([rate_hz]),
        occupancy_s=np.array([1.0]),
        effective_spike_count=np.array([1.0]),
        mark_mean=np.array([[0.0]]),
        mark_variance=np.array([[1.0]]),
        mark_feature_names=("mark0",),
        spike_mark_source="synthetic:marks",
        config=ClusterlessMarkConfig(
            mark_likelihood="diagonal-gaussian",
            mark_group_by="tetrode",
        ),
        mark_likelihood="diagonal-gaussian",
        group_ids=np.array([1]),
        group_rate_hz=np.array([[group_rate_hz]]),
        group_effective_spike_count=np.ones((1, 1)),
        group_mark_mean=np.zeros((1, 1, 1)),
        group_mark_variance=np.ones((1, 1, 1)),
    )


def _session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("synthetic"),
        position=np.array([[0.0, 0.0, 0.0, 0.0]]),
        spikes=np.empty((0, 2)),
        tetrode_cell_ids=np.empty((0, 2)),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6)),
        run_times=np.empty((0, 2)),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=np.array([0.05]),
            marks=np.array([[0.0]]),
            source_file="synthetic.mat",
            source_variable="Marks",
            feature_names=("mark0",),
            cell_ids=np.array([1]),
            group_ids=np.array([1]),
        ),
    )


def _ripple() -> RippleEvent:
    return RippleEvent(
        start=0.0,
        end=0.1,
        peak=0.05,
        raw_power=0.0,
        z_power_session=0.0,
        z_power_epoch=0.0,
    )


def test_clusterless_emissions_reject_global_rate_scale_overflow():
    encoding = _encoding(np.finfo(float).max)

    with pytest.raises(ValueError, match="scaled by spike_rate_scale must remain finite"):
        build_clusterless_mark_emissions(
            _session(),
            encoding,
            _ripple(),
            EmissionConfig(spike_rate_scale=2.0),
        )


def test_clusterless_emissions_reject_group_rate_scale_overflow():
    encoding = _encoding(1.0, group_rate_hz=np.finfo(float).max)

    with pytest.raises(ValueError, match="scaled by spike_rate_scale must remain finite"):
        build_clusterless_mark_emissions(
            _session(),
            encoding,
            _ripple(),
            EmissionConfig(spike_rate_scale=2.0),
        )
