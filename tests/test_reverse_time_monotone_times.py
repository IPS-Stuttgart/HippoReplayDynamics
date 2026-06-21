from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.result_improvement_extensions import copy_emissions_with_log_likelihood
from hipporeplayimm.reverse_models import reverse_emissions


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array([[0.0, -1.0], [-2.0, -0.5], [-0.25, -3.0]], dtype=float),
        spike_counts=np.array([[0], [2], [1]], dtype=int),
        times=np.array([0.005, 0.020, 0.055], dtype=float),
        dt=0.02,
        cell_ids=np.array([7], dtype=int),
        n_spikes=3,
        bin_durations=np.array([0.010, 0.020, 0.030], dtype=float),
        transition_durations=np.array([0.015, 0.035], dtype=float),
        metadata={"source": "unit-test"},
    )


def test_reverse_emission_helpers_keep_time_coordinates_increasing() -> None:
    emissions = _emissions()
    copied = copy_emissions_with_log_likelihood(
        emissions,
        emissions.log_likelihood,
        reverse_time=True,
    )
    reversed_emissions = reverse_emissions(emissions)

    for output in (copied, reversed_emissions):
        np.testing.assert_allclose(output.log_likelihood, emissions.log_likelihood[::-1])
        np.testing.assert_allclose(output.bin_durations, emissions.bin_durations[::-1])
        np.testing.assert_allclose(output.transition_durations, emissions.transition_durations[::-1])
        np.testing.assert_allclose(output.times, emissions.times)
        assert np.all(np.diff(output.times) > 0.0)
