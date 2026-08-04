from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm import duration_dynamics, duration_occupancy


@pytest.mark.parametrize(
    "times",
    [
        np.array([0.0, 0.0, 0.02]),
        np.array([0.0, 0.03, 0.02]),
        np.array([0.0, np.nan, 0.02]),
        np.array([0.0, np.inf, 0.02]),
    ],
)
@pytest.mark.parametrize(
    "resolver",
    [duration_dynamics.transition_durations_s, duration_occupancy.transition_durations_s],
)
def test_present_invalid_times_are_rejected(times, resolver):
    emissions = SimpleNamespace(
        n_time=3,
        dt=0.02,
        times=times,
        transition_durations=None,
    )

    with pytest.raises(ValueError, match="times must"):
        resolver(emissions)


def test_present_wrong_length_times_are_rejected():
    emissions = SimpleNamespace(
        n_time=3,
        dt=0.02,
        times=np.array([0.0, 0.02]),
        transition_durations=None,
    )

    with pytest.raises(ValueError, match="one value per emission row"):
        duration_dynamics.transition_durations_s(emissions)


def test_valid_times_still_define_transition_durations():
    emissions = SimpleNamespace(
        n_time=3,
        dt=0.02,
        times=np.array([0.005, 0.020, 0.060]),
        transition_durations=None,
    )

    np.testing.assert_allclose(
        duration_dynamics.transition_durations_s(emissions),
        np.array([0.015, 0.040]),
    )


def test_missing_times_still_fall_back_to_scalar_dt():
    emissions = SimpleNamespace(
        n_time=3,
        dt=0.02,
        times=np.array([]),
        transition_durations=None,
    )

    np.testing.assert_allclose(
        duration_dynamics.transition_durations_s(emissions),
        np.array([0.02, 0.02]),
    )
