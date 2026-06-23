from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession, RippleEvent
from hipporeplayimm.encoding import EncodingModel
from hipporeplayimm.kd_reference import KDEncodingConfig, build_kd_emissions


def _one_cell_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 4.0]),
        y_edges=np.array([0.0, 4.0]),
        bin_centers=np.array([[2.0, 2.0]]),
        rates_hz=np.array([[10.0]]),
        occupancy_s=np.array([1.0]),
        cell_ids=np.array([1]),
        config=KDEncodingConfig(),
    )


def _session_with_ripple_spikes(spikes: np.ndarray, ripple: RippleEvent) -> ReplaySession:
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


def test_build_kd_emissions_rejects_fractional_ripple_spike_cell_ids():
    ripple = RippleEvent(start=0.0, end=0.1, peak=0.05, raw_power=0.0, z_power_session=0.0, z_power_epoch=0.0)
    session = _session_with_ripple_spikes(np.array([[0.05, 1.5]]), ripple)

    with pytest.raises(ValueError, match="spike cell IDs.*integer"):
        build_kd_emissions(session, _one_cell_encoding(), ripple, time_bin_s=0.02)


def test_build_kd_emissions_rejects_boolean_encoding_cell_ids():
    ripple = RippleEvent(start=0.0, end=0.1, peak=0.05, raw_power=0.0, z_power_session=0.0, z_power_epoch=0.0)
    session = _session_with_ripple_spikes(np.array([[0.05, 1.0]]), ripple)
    encoding = _one_cell_encoding()
    encoding.cell_ids = np.array([True], dtype=bool)

    with pytest.raises(ValueError, match="encoding.cell_ids.*boolean"):
        build_kd_emissions(session, encoding, ripple, time_bin_s=0.02)
