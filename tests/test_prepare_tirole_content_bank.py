from dataclasses import replace

import numpy as np
from test_tirole_two_track import synthetic_session

from scripts.prepare_tirole_content_bank import detector_candidates


def test_detector_does_not_read_inference_or_evaluation_spikes():
    s = synthetic_session()
    times = np.r_[s.spike_times, 5 + np.arange(120) * 0.0015, 390 + np.arange(120) * 0.0015]
    units = np.r_[s.spike_units, np.arange(240) % 6]
    order = np.argsort(times)
    speed = np.where(s.sleepbox, 0, s.speed)
    s = replace(s, spike_times=times[order], spike_units=units[order], speed=speed)
    a, ma = detector_candidates(s, np.arange(6))
    assert len(a) >= 2
    assert {e["epoch"] for e in a} == {"PRE", "POST"}
    assert all(e["detector_active_cells"] == 6 for e in a)
    times = np.r_[s.spike_times, np.arange(0.5, 399, 0.01)]
    units = np.r_[s.spike_units, np.full(len(times) - len(s.spike_times), 20)]
    order = np.argsort(times)
    altered = replace(s, spike_times=times[order], spike_units=units[order])
    b, mb = detector_candidates(altered, np.arange(6))
    assert a == b and ma == mb
