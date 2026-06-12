import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import (
    EmissionConfig,
    EncodingConfig,
    EncodingModel,
    build_emissions,
    fit_place_field_encoding,
)


def test_build_emissions_maps_unsorted_encoding_cell_ids_to_their_rows():
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=np.empty((0, 4)),
        spikes=np.array([[0.25, 2.0], [0.25, 1.0]]),
        tetrode_cell_ids=np.array([[1, 1], [1, 2]]),
        excitatory_neurons=np.array([1, 2]),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.array([[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]]),
        run_times=np.empty((0, 2)),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]]),
        rates_hz=np.array([[2.0, 4.0], [8.0, 1.0]]),
        occupancy_s=np.ones(2),
        cell_ids=np.array([2, 1]),
        config=EncodingConfig(),
    )

    emissions = build_emissions(session, encoding, 0, EmissionConfig(time_bin_s=1.0))

    np.testing.assert_array_equal(emissions.spike_counts, np.array([[1, 1]]))
    expected = np.array([np.log(2.0) - 2.0 + np.log(8.0) - 8.0, np.log(4.0) - 4.0 + np.log(1.0) - 1.0])
    np.testing.assert_allclose(emissions.log_likelihood[0], expected)


def test_build_emissions_rejects_duplicate_encoding_cell_ids():
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=np.empty((0, 4)),
        spikes=np.array([[0.25, 1.0]]),
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.array([[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]]),
        run_times=np.empty((0, 2)),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]]),
        rates_hz=np.array([[2.0], [3.0]]),
        occupancy_s=np.ones(1),
        cell_ids=np.array([1, 1]),
        config=EncodingConfig(),
    )

    with pytest.raises(ValueError, match="unique"):
        build_emissions(session, encoding, 0, EmissionConfig(time_bin_s=1.0))


def test_fit_place_field_encoding_rejects_negative_min_speed(tmp_path):
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        spikes=np.empty((0, 2)),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6)),
        run_times=np.array([[0.0, 1.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )

    with pytest.raises(ValueError, match="min_speed_cm_s"):
        fit_place_field_encoding(session, EncodingConfig(min_speed_cm_s=-1.0))
