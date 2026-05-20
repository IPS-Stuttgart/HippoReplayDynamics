import numpy as np

from hipporeplayimm.duration_dynamics import (
    attach_duration_metadata,
    transition_durations_s,
)
from hipporeplayimm.encoding import LogEmissionTensor


def test_attach_duration_metadata_keeps_dt_as_plain_scalar():
    emissions = _duration_test_emissions()

    attach_duration_metadata(emissions)

    assert type(emissions.dt) is float
    assert np.isclose(emissions.dt, 0.02)
    assert np.isclose(emissions.dt * 2.0, 0.04)
    assert np.isclose(2.0 * emissions.dt, 0.04)
    assert not hasattr(emissions.dt, "transition_durations")
    np.testing.assert_allclose(emissions.transition_durations, np.array([0.015, 0.040]))


def test_transition_durations_s_prefers_explicit_metadata():
    emissions = _duration_test_emissions()
    emissions.transition_durations = np.array([0.01, 0.04])

    np.testing.assert_allclose(transition_durations_s(emissions), np.array([0.01, 0.04]))

    attach_duration_metadata(emissions)

    assert type(emissions.dt) is float
    np.testing.assert_allclose(emissions.transition_durations, np.array([0.01, 0.04]))


def _duration_test_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.70, 0.20, 0.10],
                    [0.20, 0.60, 0.20],
                    [0.10, 0.20, 0.70],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.005, 0.020, 0.060]),
        dt=0.02,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
