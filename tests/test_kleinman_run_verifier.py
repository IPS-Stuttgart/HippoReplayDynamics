import numpy as np

from scripts.validate_kleinman_run_decoder import fit_maps, interval_counts
from scripts.verify_kleinman_run_decoder import independent_map


def test_raw_spike_histogram_matches_frame_count_map():
    rng = np.random.default_rng(7)
    t = np.arange(0, 20.01, .05)
    trains = [np.sort(rng.uniform(-1, 21, 700)) for _ in range(9)]
    trains[0] = np.sort(np.r_[trains[0], t[0], t[5], t[-1]])
    d = np.repeat([0, 1], 200)
    b = np.tile(np.arange(20), 20)
    mask = np.arange(400) % 4 != 0
    ordinary = fit_maps(interval_counts(trains, t), np.diff(t), d, b, mask, 20)
    independent = independent_map(trains, t, np.diff(t), d, b, mask, 20)
    for a, b in zip(ordinary[:3], independent, strict=True):
        np.testing.assert_allclose(a, b)
