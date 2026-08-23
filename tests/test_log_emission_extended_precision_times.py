from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _extended_precision_times() -> np.ndarray:
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("np.longdouble does not provide precision beyond float64")

    start = np.longdouble(2) ** 60
    return np.array(
        [
            start,
            start + np.longdouble(1),
            start + np.longdouble(3),
        ],
        dtype=np.longdouble,
    )


def test_log_emission_preserves_wide_times_before_duration_differencing() -> None:
    times = _extended_precision_times()
    narrowed = np.asarray(times, dtype=float)
    assert np.unique(narrowed).size == 1

    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((3, 1), dtype=float),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=times,
        dt=1.0,
        cell_ids=np.array([7], dtype=int),
        n_spikes=0,
    )

    assert emissions.times.dtype == times.dtype
    np.testing.assert_array_equal(emissions.times, times)
    np.testing.assert_array_equal(
        emissions.transition_durations,
        np.array([1.0, 2.0], dtype=float),
    )
