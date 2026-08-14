from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _emissions(*, times: np.ndarray, transition_durations: np.ndarray) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((times.size, 2), dtype=float),
        spike_counts=np.zeros((times.size, 1), dtype=int),
        times=times,
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
        bin_durations=np.full(times.size, 0.02, dtype=float),
        transition_durations=transition_durations,
    )


@pytest.mark.parametrize(
    "times",
    [
        np.array([0.01, 0.01, 0.03]),
        np.array([0.01, 0.04, 0.03]),
    ],
)
def test_explicit_transition_durations_do_not_bypass_timestamp_order(times: np.ndarray) -> None:
    with pytest.raises(ValueError, match="times must be strictly increasing"):
        _emissions(times=times, transition_durations=np.array([0.02, 0.02]))


def test_explicit_transition_durations_preserve_valid_increasing_timestamps() -> None:
    emissions = _emissions(
        times=np.array([0.01, 0.03, 0.07]),
        transition_durations=np.array([0.02, 0.04]),
    )

    np.testing.assert_allclose(emissions.times, np.array([0.01, 0.03, 0.07]))
    np.testing.assert_allclose(emissions.transition_durations, np.array([0.02, 0.04]))
