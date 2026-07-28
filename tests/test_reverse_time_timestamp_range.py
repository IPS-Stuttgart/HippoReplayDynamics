from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.result_improvement_extensions import copy_emissions_with_log_likelihood
from hipporeplayimm.reverse_models import reverse_emissions


def _emissions(
    times: np.ndarray,
    transition_durations: np.ndarray,
) -> LogEmissionTensor:
    n_time = int(times.size)
    return LogEmissionTensor(
        log_likelihood=np.zeros((n_time, 1), dtype=float),
        spike_counts=np.zeros((n_time, 1), dtype=int),
        times=times,
        dt=1.0,
        cell_ids=np.array([7], dtype=int),
        n_spikes=0,
        bin_durations=np.ones(n_time, dtype=float),
        transition_durations=transition_durations,
    )


def _reverse_with_path(emissions: LogEmissionTensor, path: str) -> LogEmissionTensor:
    if path == "copy":
        return copy_emissions_with_log_likelihood(
            emissions,
            emissions.log_likelihood,
            reverse_time=True,
        )
    return reverse_emissions(emissions)


@pytest.mark.parametrize("path", ["copy", "reverse"])
def test_reverse_rejects_timestamp_increment_below_float_resolution(path: str) -> None:
    start = 1e308
    emissions = _emissions(
        np.array([start, np.nextafter(start, np.inf)], dtype=float),
        np.array([1.0], dtype=float),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        _reverse_with_path(emissions, path)


@pytest.mark.parametrize("path", ["copy", "reverse"])
def test_reverse_rejects_cumulative_timestamp_overflow(path: str) -> None:
    emissions = _emissions(
        np.array([0.0, 1.0, 2.0], dtype=float),
        np.array([1e308, 1e308], dtype=float),
    )

    with pytest.raises(ValueError, match="floating-point range"):
        _reverse_with_path(emissions, path)
