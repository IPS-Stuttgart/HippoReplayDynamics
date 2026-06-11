import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import (
    EncodingConfig,
    EmissionConfig,
    EncodingModel,
    build_emissions,
    fit_place_field_encoding,
    _positions_to_flat_bins,
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


def test_fit_place_field_encoding_falls_back_to_all_spikes_without_excitatory_labels(tmp_path):
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
        excitatory_neurons=np.array([], dtype=int),
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

    assert encoding.cell_ids.tolist() == [1]
    assert encoding.rates_hz[0].max() > encoding.config.rate_floor_hz
    assert 40.0 <= peak[0] <= 60.0


@pytest.mark.parametrize(
    "position",
    [
        np.empty((0, 4)),
        np.array([[0.0, 0.0, 0.0, 0.0]]),
        np.array(
            [
                [0.0, np.nan, 0.0, 0.0],
                [1.0, 1.0, np.nan, 0.0],
            ]
        ),
    ],
)
def test_fit_place_field_encoding_rejects_empty_or_too_short_position(position):
    session = _single_ripple_session()
    session.position = position

    with pytest.raises(ValueError, match="at least two finite position samples"):
        fit_place_field_encoding(session)


@pytest.mark.parametrize(
    "times",
    [
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 1.0, 0.5]),
    ],
)
def test_fit_place_field_encoding_rejects_nonincreasing_position_times(times):
    session = _single_ripple_session()
    session.position = np.column_stack([times, np.array([0.0, 1.0, 2.0]), np.zeros(3), np.zeros(3)])

    with pytest.raises(ValueError, match="strictly increasing"):
        fit_place_field_encoding(session)


def test_fit_place_field_encoding_handles_empty_cell_set_with_smoothing(tmp_path):
    times = np.linspace(0.0, 10.0, 301)
    x = np.linspace(0.0, 100.0, times.size)
    y = np.zeros_like(x)
    position = np.column_stack([times, x, y, np.zeros_like(x)])
    session = ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=position,
        spikes=np.empty((0, 2)),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
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
        EncodingConfig(bin_size_cm=10.0, smoothing_sigma_bins=1.0, min_speed_cm_s=1.0),
    )

    assert encoding.cell_ids.size == 0
    assert encoding.rates_hz.shape == (0, encoding.n_bins)
    assert encoding.occupancy_s.shape == (encoding.n_bins,)


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


def test_fit_place_field_encoding_handles_no_selected_cells_with_smoothing(tmp_path):
    times = np.linspace(0.0, 1.0, 11)
    x = np.linspace(0.0, 10.0, times.size)
    y = np.zeros_like(x)
    session = _session_without_spikes(np.column_stack([times, x, y, np.zeros_like(x)]), tmp_path)

    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(bin_size_cm=5.0, smoothing_sigma_bins=1.0, min_speed_cm_s=0.0),
    )

    assert encoding.cell_ids.size == 0
    assert encoding.rates_hz.shape == (0, encoding.n_bins)


def test_fit_place_field_encoding_handles_single_position_sample_without_spikes(tmp_path):
    session = _session_without_spikes(
        np.array([[0.0, 2.0, 3.0, 0.0]]),
        tmp_path,
    )

    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(bin_size_cm=5.0, smoothing_sigma_bins=0.0, min_speed_cm_s=0.0),
    )

    assert encoding.n_bins > 0
    assert encoding.cell_ids.size == 0
    assert encoding.rates_hz.shape == (0, encoding.n_bins)


def test_fit_place_field_encoding_rejects_empty_position_samples(tmp_path):
    session = _session_without_spikes(np.empty((0, 4)), tmp_path)

    with pytest.raises(ValueError, match="finite position"):
        fit_place_field_encoding(session)


def test_select_cells_keeps_cell_ids_sorted_for_searchsorted():
    encoding = fit_place_field_encoding(
        _linear_session_with_two_cells(),
        EncodingConfig(bin_size_cm=10.0, smoothing_sigma_bins=0.0, min_speed_cm_s=1.0),
    )

    selected = encoding.select_cells([2, 1])

    assert selected.cell_ids.tolist() == [1, 2]


def test_select_cells_rejects_missing_cell_ids_with_clear_error():
    encoding = fit_place_field_encoding(
        _linear_session_with_two_cells(),
        EncodingConfig(bin_size_cm=10.0, smoothing_sigma_bins=0.0, min_speed_cm_s=1.0),
    )

    with pytest.raises(ValueError, match="99"):
        encoding.select_cells([1, 99])


def test_positions_to_flat_bins_includes_closed_upper_grid_edge():
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0, 2.0]),
        bin_centers=np.array(
            [
                [0.5, 0.5],
                [0.5, 1.5],
                [1.5, 0.5],
                [1.5, 1.5],
            ]
        ),
        rates_hz=np.ones((1, 4), dtype=float),
        occupancy_s=np.ones(4, dtype=float),
        cell_ids=np.array([1]),
        config=EncodingConfig(),
    )
    xy = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [2.0, 0.5],
            [0.5, 2.0],
            [2.0 + 1e-6, 0.5],
            [0.5, 2.0 + 1e-6],
        ]
    )
    expected = np.array([0, 3, 2, 1, -1, -1])

    np.testing.assert_array_equal(encoding.positions_to_flat_bins(xy), expected)
    np.testing.assert_array_equal(_positions_to_flat_bins(xy, encoding.x_edges, encoding.y_edges), expected)


def test_build_emissions_applies_spike_rate_scale_to_expected_counts():
    session = _single_ripple_session()
    encoding = _two_bin_encoding()

    emissions = build_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=1.0, spike_rate_scale=3.0),
    )

    expected = np.array([2.0, 4.0]) * 3.0
    np.testing.assert_allclose(emissions.log_likelihood[0], np.log(expected) - expected)


def test_build_emissions_applies_likelihood_temperature():
    session = _single_ripple_session()
    encoding = _two_bin_encoding()

    base = build_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=1.0),
    )
    tempered = build_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=1.0, likelihood_temperature=2.0),
    )

    np.testing.assert_allclose(tempered.log_likelihood, base.log_likelihood / 2.0)


def test_build_emissions_applies_cell_weights_to_cell_log_terms():
    session = _two_cell_ripple_session()
    encoding = _two_cell_two_bin_encoding()

    emissions = build_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=1.0, cell_weights=[1.0, 0.0]),
    )

    expected = np.log(np.array([2.0, 4.0])) - np.array([2.0, 4.0])
    np.testing.assert_allclose(emissions.log_likelihood[0], expected)


def test_build_emissions_accepts_zero_dimensional_numpy_cell_weight():
    session = _single_ripple_session()
    encoding = _two_bin_encoding()

    expected = build_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=1.0, cell_weights=[1.0]),
    )
    emissions = build_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=1.0, cell_weights=np.array(1.0)),
    )

    np.testing.assert_allclose(emissions.log_likelihood, expected.log_likelihood)


@pytest.mark.parametrize("cell_weights", ([1.0, 1.0], [-1.0], [0.0], [np.nan]))
def test_build_emissions_rejects_invalid_cell_weights(cell_weights):
    with pytest.raises(ValueError, match="cell_weights"):
        build_emissions(
            _single_ripple_session(),
            _two_bin_encoding(),
            0,
            EmissionConfig(time_bin_s=1.0, cell_weights=cell_weights),
        )


def test_build_emissions_supports_negative_binomial_overdispersion():
    session = _single_ripple_session()
    encoding = _two_bin_encoding()

    emissions = build_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=1.0, negative_binomial_overdispersion=0.5),
    )

    mean = np.array([2.0, 4.0])
    size = 1.0 / 0.5
    expected = (
        np.log(size)
        + size * (np.log(size) - np.log(size + mean))
        + np.log(mean) - np.log(size + mean)
    )
    np.testing.assert_allclose(emissions.log_likelihood[0], expected)


def test_build_emissions_rejects_invalid_emission_calibration():
    with pytest.raises(ValueError, match="likelihood_temperature"):
        build_emissions(_single_ripple_session(), _two_bin_encoding(), 0, EmissionConfig(time_bin_s=1.0, likelihood_temperature=0.0))
    with pytest.raises(ValueError, match="negative_binomial_overdispersion"):
        build_emissions(_single_ripple_session(), _two_bin_encoding(), 0, EmissionConfig(time_bin_s=1.0, negative_binomial_overdispersion=-0.1))


def test_build_emissions_rejects_nonpositive_spike_rate_scale():
    with pytest.raises(ValueError, match="spike_rate_scale"):
        build_emissions(
            _single_ripple_session(),
            _two_bin_encoding(),
            0,
            EmissionConfig(time_bin_s=1.0, spike_rate_scale=0.0),
        )


def test_build_emissions_rejects_nonpositive_likelihood_temperature():
    with pytest.raises(ValueError, match="likelihood_temperature"):
        build_emissions(
            _single_ripple_session(),
            _two_bin_encoding(),
            0,
            EmissionConfig(time_bin_s=1.0, likelihood_temperature=0.0),
        )


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


def _single_ripple_session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        spikes=np.array([[0.5, 1.0]]),
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]]),
        run_times=np.array([[0.0, 1.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


def _two_cell_ripple_session() -> ReplaySession:
    session = _single_ripple_session()
    return ReplaySession(
        rat=session.rat,
        name=session.name,
        path=session.path,
        position=session.position,
        spikes=np.array([[0.5, 1.0], [0.5, 2.0]]),
        tetrode_cell_ids=np.array([[1, 1], [1, 2]]),
        excitatory_neurons=np.array([1, 2]),
        inhibitory_neurons=np.array([]),
        ripple_events=session.ripple_events,
        run_times=session.run_times,
        sleep_box_immobile_times=session.sleep_box_immobile_times,
        sleep_times=session.sleep_times,
        rem_times=session.rem_times,
        well_sequence=session.well_sequence,
        metadata=session.metadata,
    )


def _two_bin_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]]),
        rates_hz=np.array([[2.0, 4.0]]),
        occupancy_s=np.ones(2),
        cell_ids=np.array([1]),
        config=EncodingConfig(),
    )


def _two_cell_two_bin_encoding() -> EncodingModel:
    encoding = _two_bin_encoding()
    return EncodingModel(
        x_edges=encoding.x_edges,
        y_edges=encoding.y_edges,
        bin_centers=encoding.bin_centers,
        rates_hz=np.array([[2.0, 4.0], [8.0, 1.0]]),
        occupancy_s=encoding.occupancy_s,
        cell_ids=np.array([1, 2]),
        config=encoding.config,
    )


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


def _session_without_spikes(position: np.ndarray, path) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=path,
        position=position,
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
