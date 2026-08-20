import numpy as np

from hipporeplayimm.clusterless import (
    ClusterlessMarkConfig,
    fit_clusterless_mark_encoding,
)
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import (
    EncodingConfig,
    _interp_positions,
    fit_place_field_encoding,
)


def test_interp_positions_does_not_clamp_queries_outside_tracking_support():
    times = np.array([1.0, 2.0, 3.0])
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])

    interpolated = _interp_positions(
        times,
        xy,
        np.array([0.5, 1.0, 1.5, 3.0, 3.5]),
    )

    assert np.isnan(interpolated[0]).all()
    assert np.allclose(interpolated[1], [0.0, 0.0])
    assert np.allclose(interpolated[2], [5.0, 0.0])
    assert np.allclose(interpolated[3], [20.0, 0.0])
    assert np.isnan(interpolated[4]).all()


def test_place_field_training_ignores_spikes_outside_position_time_support(tmp_path):
    config = _encoding_config()
    baseline = fit_place_field_encoding(
        _training_session(tmp_path, include_outside_events=False),
        config,
    )
    with_outside_events = fit_place_field_encoding(
        _training_session(tmp_path, include_outside_events=True),
        config,
    )

    assert baseline.rates_hz[0, 0] == config.rate_floor_hz
    assert baseline.rates_hz[0, 1] > config.rate_floor_hz
    assert np.allclose(with_outside_events.rates_hz, baseline.rates_hz)
    assert np.allclose(with_outside_events.occupancy_s, baseline.occupancy_s)


def test_clusterless_training_ignores_marks_outside_position_time_support(tmp_path):
    clusterless_config = ClusterlessMarkConfig(
        encoding=_encoding_config(),
        mark_likelihood="diagonal-gaussian",
        mark_smoothing_sigma_bins=0.0,
        mark_prior_count=1.0,
        mark_variance_floor=0.1,
        rate_floor_hz=1e-6,
    )
    baseline = fit_clusterless_mark_encoding(
        _training_session(tmp_path, include_outside_events=False),
        clusterless_config,
    )
    with_outside_events = fit_clusterless_mark_encoding(
        _training_session(tmp_path, include_outside_events=True),
        clusterless_config,
    )

    assert np.allclose(with_outside_events.rate_hz, baseline.rate_hz)
    assert np.allclose(
        with_outside_events.effective_spike_count,
        baseline.effective_spike_count,
    )
    assert np.allclose(with_outside_events.mark_mean, baseline.mark_mean)
    assert np.allclose(with_outside_events.mark_variance, baseline.mark_variance)


def _encoding_config() -> EncodingConfig:
    return EncodingConfig(
        bin_size_cm=10.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=0.0,
        min_occupancy_s=0.01,
        rate_floor_hz=1e-6,
        arena_padding_cm=0.0,
        exclude_ripple_intervals=False,
    )


def _training_session(tmp_path, *, include_outside_events: bool) -> ReplaySession:
    position_times = np.array([1.0, 2.0, 3.0])
    position = np.column_stack(
        [
            position_times,
            np.array([0.0, 10.0, 20.0]),
            np.zeros(position_times.shape[0]),
            np.zeros(position_times.shape[0]),
        ]
    )
    if include_outside_events:
        event_times = np.array([0.5, 2.0, 3.5])
        mark_values = np.array([[-100.0], [5.0], [100.0]])
    else:
        event_times = np.array([2.0])
        mark_values = np.array([[5.0]])
    cell_ids = np.ones(event_times.shape[0], dtype=int)
    spikes = np.column_stack([event_times, cell_ids])

    return ReplaySession(
        rat="RatX",
        name="OpenPositionSupport",
        path=tmp_path,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 4.0]]),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=event_times,
            marks=mark_values,
            source_file="Spike_Data.mat",
            source_variable="Spike_Amplitude_Marks",
            feature_names=("amp",),
            cell_ids=cell_ids,
        ),
    )
