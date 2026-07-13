from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm  # noqa: F401  # import applies runtime patches
from hipporeplayimm import ground_truth
from hipporeplayimm.benchmark_event_limit_validation import _coerce_optional_nonnegative_integer
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


def test_event_limit_validation_preserves_large_integer_counts_exactly() -> None:
    large = 2**53 + 1

    assert _coerce_optional_nonnegative_integer(large, "max_events_per_session") == large
    assert _coerce_optional_nonnegative_integer(str(large), "max_events_per_session") == large
    assert (
        _coerce_optional_nonnegative_integer(
            np.array(large, dtype=object),
            "max_events_per_session",
        )
        == large
    )


def test_event_indices_preserve_unlimited_event_selection() -> None:
    indices = _event_indices(
        _SessionWithRunRipples(),
        BenchmarkConfig(max_events_per_session=None),
    )

    assert indices.tolist() == [0, 1, 2]


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
def test_ground_truth_generator_rejects_invalid_event_limits_before_loading(
    monkeypatch,
    bad_limit,
) -> None:
    def unexpected_load(_root):
        raise AssertionError("dataset loading must not start for an invalid event limit")

    monkeypatch.setattr(ground_truth, "load_open_field_sessions", unexpected_load)

    with pytest.raises(ValueError, match="max_events_per_session"):
        ground_truth.generate_behavioral_ground_truth(
            ".",
            max_events_per_session=bad_limit,
        )


def test_ground_truth_generator_accepts_integer_valued_event_limit(monkeypatch) -> None:
    sessions = [object(), object()]
    monkeypatch.setattr(ground_truth, "load_open_field_sessions", lambda _root: sessions)
    monkeypatch.setattr(
        ground_truth,
        "label_session_behavioral_ground_truth",
        lambda _session, _config: pd.DataFrame({"event_index": [0, 1, 2]}),
    )

    rows = ground_truth.generate_behavioral_ground_truth(
        ".",
        max_events_per_session="1",
    )

    assert rows["event_index"].tolist() == [0, 0]
