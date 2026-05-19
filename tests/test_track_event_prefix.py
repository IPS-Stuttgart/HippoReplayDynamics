import numpy as np
import pytest

from hipporeplayimm.duration_dynamics import attach_duration_metadata, transition_durations_s
from hipporeplayimm.encoding import LogEmissionTensor
from scripts.track_event import _prefix_emissions


def _duration_annotated_emissions() -> LogEmissionTensor:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.2, 0.3, 0.5],
                ]
            )
        ),
        spike_counts=np.zeros((4, 1), dtype=int),
        times=np.array([0.00, 0.02, 0.05, 0.09]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    return attach_duration_metadata(emissions)


def test_prefix_emissions_slices_duration_metadata_for_shorter_prefixes():
    emissions = _duration_annotated_emissions()

    prefix = _prefix_emissions(emissions, 2)

    assert prefix.n_time == 2
    assert float(prefix.dt) == pytest.approx(0.02)
    assert np.allclose(transition_durations_s(prefix), np.array([0.02]))


def test_prefix_emissions_handles_single_bin_prefix_with_empty_durations():
    emissions = _duration_annotated_emissions()

    prefix = _prefix_emissions(emissions, 1)

    assert prefix.n_time == 1
    assert transition_durations_s(prefix).shape == (0,)
