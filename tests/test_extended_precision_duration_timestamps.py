from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm import duration_dynamics, duration_occupancy


@pytest.mark.parametrize(
    "resolver",
    [
        duration_dynamics.transition_durations_s,
        duration_occupancy.transition_durations_s,
    ],
)
def test_extended_precision_timestamps_do_not_collapse_before_differencing(
    resolver,
) -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("platform longdouble has no precision beyond binary64")

    base = np.longdouble(2**53)
    emissions = SimpleNamespace(
        n_time=2,
        dt=7.0,
        times=np.array(
            [base, base + np.longdouble(1)],
            dtype=np.longdouble,
        ),
        transition_durations=None,
    )

    np.testing.assert_array_equal(
        resolver(emissions),
        np.array([1.0], dtype=float),
    )
