from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm  # noqa: F401  # import applies runtime patches
from hipporeplayimm.benchmarks import BenchmarkConfig, _event_indices


class _SessionWithRunRipples:
    ripple_count = 3

    def ripple_indices_in_run(self) -> np.ndarray:
        return np.array([0, 1, 2], dtype=int)


@pytest.mark.parametrize(
    "bad_limit",
    [
        True,
        False,
        np.bool_(True),
        np.array(False),
        1.5,
        "2.5",
        -1,
        float("nan"),
        float("inf"),
        np.array([1]),
    ],
)
def test_event_indices_reject_invalid_max_events_per_session(bad_limit) -> None:
    with pytest.raises(ValueError, match="max_events_per_session"):
        _event_indices(
            _SessionWithRunRipples(),
            BenchmarkConfig(max_events_per_session=bad_limit),
        )


@pytest.mark.parametrize("limit", [1, 1.0, np.int64(1), np.array(1.0), "1"])
def test_event_indices_accept_integer_valued_event_limits(limit) -> None:
    indices = _event_indices(
        _SessionWithRunRipples(),
        BenchmarkConfig(max_events_per_session=limit),
    )

    assert indices.tolist() == [0]


def test_event_indices_preserve_unlimited_event_selection() -> None:
    indices = _event_indices(
        _SessionWithRunRipples(),
        BenchmarkConfig(max_events_per_session=None),
    )

    assert indices.tolist() == [0, 1, 2]
