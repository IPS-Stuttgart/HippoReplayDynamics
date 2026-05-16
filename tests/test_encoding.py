import numpy as np

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import (
    EncodingConfig,
    EmissionConfig,
    EncodingModel,
    build_emissions,
    fit_place_field_encoding,
)


def test_fit_place_field_encoding_recovers_peak_near_spike_location(tmp_path):
    times = np.linspace(0.0, 10.0, 301)
    x = np.linspace(0.0, 100.0, times.size)
    y = np.zeros_like(x)
    position = np.column_stack([times, x, y, np.zeros_like(x)])
    spike_times = times[(x > 45.0) & (x < 55.0)][::2]
    spikes = np.column_stack([spike_times, np.ones(spike_times.shape)])
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.empty((0, 6)),
        run_times=np.array([[0.0, 10.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )

    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(bin_size_cm=5.0, smoothing_sigma_bins=0.0, min_speed_cm_s=1.0),
    )
    peak = encoding.bin_centers[int(np.argmax(encoding.rates_hz[0]))]

    assert 40.0 <= peak[0] <= 60.0


def test_fit_place_field_encoding_excludes_ripple_intervals_by_default(tmp_path):
    session = _linear_session_with_ripple_spikes(tmp_path)
    config = EncodingConfig(bin_size_cm=10.0, smoothing_sigma_bins=0.0, min_speed_cm_s=1.0)

    encoding = fit_place_field_encoding(session, config)
    peak = encoding.bin_centers[int(np.argmax(encoding.rates_hz[0]))]

    assert peak[0] < 35.0


def test_fit_place_field_encoding_can_include_ripples_for_legacy_reproduction(tmp_path):
    session = _linear_session_with_ripple_spikes(tmp_path)
    config = EncodingConfig(
        bin_size_cm=10.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=1.0,
        exclude_ripple_intervals=False,
    )

    encoding = fit_place_field_encoding(session, config)
    peak = encoding.bin_centers[int(np.argmax(encoding.rates_hz[0]))]

    assert peak[0] > 65.0


def test_select_cells_keeps_cell_ids_sorted_for_searchsorted():
    encoding = fit_place_field_encoding(
        _linear_session_with_two_cells(),
        EncodingConfig(bin_size_cm=10.0, smoothing_sigma_bins=0.0, min_speed_cm_s=1.0),
    )

    selected = encoding.select_cells([2, 1])

    assert selected.cell_ids.tolist() == [1, 2]


def test_build_emissions_clips_to_ripple_end_and_uses_partial_bin_duration():
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=np.empty((0, 4)),
        spikes=np.array([[0.049, 1.0], [0.054, 1.0]]),
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[0.0, 0.053, 0.0265, 0.0, 0.0, 0.0]]),
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
        rates_hz=np.array([[10.0]]),
        occupancy_s=np.array([1.0]),
        cell_ids=np.array([1]),
        config=EncodingConfig(),
    )

    emissions = build_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=0.02),
    )

    assert emissions.spike_counts[:, 0].tolist() == [0, 0, 1]
    assert emissions.n_spikes == 1
    assert np.allclose(emissions.times, np.array([0.01, 0.03, 0.0465]))
    assert np.isclose(emissions.dt, 0.02)

    expected_log_likelihood = np.array([-0.2, -0.2, np.log(0.13) - 0.13])
    assert np.allclose(emissions.log_likelihood[:, 0], expected_log_likelihood)


def _linear_session_with_two_cells():
    times = np.linspace(0.0, 10.0, 301)
    x = np.linspace(0.0, 100.0, times.size)
    y = np.zeros_like(x)
    position = np.column_stack([times, x, y, np.zeros_like(x)])
    spikes = np.array([[2.0, 1.0], [5.0, 2.0]])
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1], [1, 2]]),
        excitatory_neurons=np.array([1, 2]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.empty((0, 6)),
        run_times=np.array([[0.0, 10.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


def _linear_session_with_ripple_spikes(path):
    times = np.linspace(0.0, 10.0, 301)
    x = np.linspace(0.0, 100.0, times.size)
    y = np.zeros_like(x)
    position = np.column_stack([times, x, y, np.zeros_like(x)])
    non_ripple_spikes = np.array([[2.0, 1.0], [2.1, 1.0]])
    ripple_spikes = np.column_stack([np.linspace(7.2, 7.8, 6), np.ones(6)])
    spikes = np.vstack([non_ripple_spikes, ripple_spikes])
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=path,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[7.0, 8.0, 7.5, 0.0, 0.0, 0.0]]),
        run_times=np.array([[0.0, 10.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )
