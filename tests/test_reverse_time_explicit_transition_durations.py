from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.result_improvement_extensions import copy_emissions_with_log_likelihood
from hipporeplayimm.reverse_models import reverse_emissions


def test_reverse_emission_times_follow_explicit_transition_durations() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array([[0.0], [-0.25], [-0.5]], dtype=float),
        spike_counts=np.array([[0], [1], [0]], dtype=int),
        times=np.array([10.0, 10.2, 10.7], dtype=float),
        dt=0.2,
        cell_ids=np.array([7], dtype=int),
        n_spikes=1,
        bin_durations=np.array([0.2, 0.2, 0.2], dtype=float),
        transition_durations=np.array([0.1, 0.3], dtype=float),
    )

    expected_transition_durations = emissions.transition_durations[::-1]
    expected_times = float(emissions.times[0]) + np.concatenate(
        ([0.0], np.cumsum(expected_transition_durations))
    )

    copied = copy_emissions_with_log_likelihood(
        emissions,
        emissions.log_likelihood,
        reverse_time=True,
    )
    reversed_emissions = reverse_emissions(emissions)

    for output in (copied, reversed_emissions):
        np.testing.assert_allclose(output.log_likelihood, emissions.log_likelihood[::-1])
        np.testing.assert_allclose(output.transition_durations, expected_transition_durations)
        np.testing.assert_allclose(output.times, expected_times)
        np.testing.assert_allclose(np.diff(output.times), output.transition_durations)
        assert np.all(np.diff(output.times) > 0.0)
