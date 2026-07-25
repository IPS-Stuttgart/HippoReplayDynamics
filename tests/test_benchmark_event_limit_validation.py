from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm  # noqa: F401  # import applies runtime patches
from hipporeplayimm.benchmark_event_limit_validation import (
    _coerce_boolean_scalar,
    _coerce_optional_nonnegative_integer,
)
from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    _benchmark_config_metadata,
    _benchmark_split_metadata,
    _event_indices,
)


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
    "bad_flag",
    [
        "false",
        "true",
        0,
        1,
        None,
        np.array([False]),
        np.array(False, dtype=object),
    ],
)
def test_event_indices_reject_nonboolean_randomize_event_subset(bad_flag) -> None:
    with pytest.raises(ValueError, match="randomize_event_subset must be a boolean scalar"):
        _event_indices(
            _SessionWithRunRipples(),
            BenchmarkConfig(
                max_events_per_session=2,
                randomize_event_subset=bad_flag,
            ),
        )


@pytest.mark.parametrize(
    "flag",
    [
        False,
        True,
        np.bool_(False),
        np.bool_(True),
        np.array(False),
        np.array(True),
    ],
)
def test_event_indices_accept_boolean_randomize_event_subset(flag) -> None:
    indices = _event_indices(
        _SessionWithRunRipples(),
        BenchmarkConfig(
            max_events_per_session=2,
            randomize_event_subset=flag,
            event_subset_seed=7,
        ),
    )

    assert indices.shape == (2,)
    assert np.array_equal(indices, np.sort(indices))


def test_randomize_event_subset_boolean_coercion_is_strict() -> None:
    assert _coerce_boolean_scalar(np.array(True), "randomize_event_subset") is True
    assert _coerce_boolean_scalar(np.bool_(False), "randomize_event_subset") is False

    with pytest.raises(ValueError, match="boolean scalar"):
        _coerce_boolean_scalar("false", "randomize_event_subset")


def test_benchmark_metadata_rejects_nonboolean_randomize_event_subset() -> None:
    config = BenchmarkConfig(randomize_event_subset="false")

    with pytest.raises(ValueError, match="randomize_event_subset must be a boolean scalar"):
        _benchmark_config_metadata(config)
    with pytest.raises(ValueError, match="randomize_event_subset must be a boolean scalar"):
        _benchmark_split_metadata(config, 0)


def test_benchmark_metadata_uses_canonical_randomize_boolean() -> None:
    config = BenchmarkConfig(randomize_event_subset=np.array(True))

    config_metadata = _benchmark_config_metadata(config)
    split_metadata = _benchmark_split_metadata(config, 0)

    assert config_metadata["benchmark_randomize_event_subset"] is True
    assert split_metadata["benchmark_randomize_event_subset"] is True
