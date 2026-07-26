from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.ground_truth import active_goal_at_time


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        well_sequence=np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
    )


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (np.nan, ValueError),
        (np.inf, ValueError),
        (-np.inf, ValueError),
        (True, TypeError),
        ("1.5", TypeError),
        (np.array([1.5]), TypeError),
    ],
)
def test_active_goal_rejects_invalid_direct_timestamps(
    value: object,
    error: type[Exception],
):
    with pytest.raises(error, match="time_s"):
        active_goal_at_time(_session(), value)


def test_active_goal_accepts_numeric_scalar_wrappers():
    session = _session()

    assert active_goal_at_time(session, np.array(1.5)) == 1
    assert active_goal_at_time(session, np.float32(2.0)) == 2
