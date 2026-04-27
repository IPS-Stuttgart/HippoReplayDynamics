import numpy as np

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, fit_place_field_encoding


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
