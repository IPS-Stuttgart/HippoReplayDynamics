from __future__ import annotations

from pathlib import Path

import numpy as np

from hipporeplayimm.data import ReplaySession, RippleEvent
from hipporeplayimm.encoding import EncodingModel
from hipporeplayimm.kd_reference import KDEncodingConfig, build_kd_emissions, poisson_log_emissions


def test_build_kd_emissions_maps_unsorted_encoding_cell_ids_by_identifier() -> None:
    ripple = RippleEvent(
        start=0.0,
        end=0.02,
        peak=0.01,
        raw_power=0.0,
        z_power_session=0.0,
        z_power_epoch=0.0,
    )
    session = ReplaySession(
        rat="rat",
        name="session",
        path=Path("."),
        position=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        spikes=np.array([[0.005, 7.0], [0.015, 2.0]]),
        tetrode_cell_ids=np.empty((0, 2)),
        excitatory_neurons=np.array([], dtype=int),
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
    encoding = EncodingModel(
        x_edges=np.array([0.0, 4.0]),
        y_edges=np.array([0.0, 4.0]),
        bin_centers=np.array([[2.0, 2.0]]),
        rates_hz=np.array([[10.0], [20.0]]),
        occupancy_s=np.array([1.0]),
        cell_ids=np.array([7, 2]),
        config=KDEncodingConfig(),
    )

    emissions = build_kd_emissions(session, encoding, ripple, time_bin_s=0.02)

    np.testing.assert_array_equal(emissions.spike_counts, np.array([[1, 1]]))
    assert emissions.n_spikes == 2
    expected_log_likelihood = poisson_log_emissions(
        np.array([[1, 1]]),
        encoding.rates_hz,
        np.array([0.02]),
    )
    np.testing.assert_allclose(emissions.log_likelihood, expected_log_likelihood)
