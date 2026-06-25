from pathlib import Path

import numpy as np

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.kd_reference import KDEncodingConfig, fit_kd_place_field_encoding


def test_kd_place_field_encoding_falls_back_to_all_spikes_without_excitatory_labels():
    times = np.linspace(0.0, 10.0, 301)
    x = np.linspace(0.0, 100.0, times.size)
    y = np.zeros_like(x)
    position = np.column_stack([times, x, y, np.zeros_like(x)])
    spike_times = times[(x > 45.0) & (x < 55.0)][::2]
    spikes = np.column_stack([spike_times, np.ones(spike_times.shape)])
    session = ReplaySession(
        rat="rat",
        name="session",
        path=Path("."),
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1]]),
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

    encoding = fit_kd_place_field_encoding(
        session,
        KDEncodingConfig(
            bin_size_cm=5.0,
            n_bins_x=24,
            n_bins_y=2,
            smoothing_sigma_cm=0.0,
            min_speed_cm_s=1.0,
            min_peak_rate_hz=0.0,
        ),
    )

    peak = encoding.bin_centers[int(np.argmax(encoding.rates_hz[0]))]

    assert encoding.cell_ids.tolist() == [1]
    assert encoding.rates_hz[0].max() > encoding.config.rate_floor_hz
    assert 40.0 <= peak[0] <= 60.0
