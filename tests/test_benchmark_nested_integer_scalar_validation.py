from __future__ import annotations

import warnings

import numpy as np
import pytest

import hipporeplayimm  # noqa: F401  # import applies runtime patches
from hipporeplayimm import benchmarks
from hipporeplayimm import benchmark_event_limit_validation as event_limits
from hipporeplayimm import benchmark_seed_validation as seed_validation


class _SessionWithRunRipples:
    ripple_count = 3

    def ripple_indices_in_run(self) -> np.ndarray:
        return np.array([0, 1, 2], dtype=int)


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def test_event_limit_rejects_nested_boolean_scalar() -> None:
    with pytest.raises(ValueError, match="max_events_per_session"):
        benchmarks._event_indices(
            _SessionWithRunRipples(),
            benchmarks.BenchmarkConfig(
                max_events_per_session=_nested_object_scalar(np.bool_(True)),
            ),
        )


def test_cell_split_count_rejects_nested_boolean_scalar() -> None:
    with pytest.raises(ValueError, match="n_cell_splits"):
        benchmarks._n_cell_splits(
            benchmarks.BenchmarkConfig(
                n_cell_splits=_nested_object_scalar(True),
            )
        )


def test_seed_validation_rejects_nested_boolean_scalar() -> None:
    with pytest.raises(ValueError, match="random_seed"):
        benchmarks._cell_split_seed(_nested_object_scalar(True), 0)


def test_nested_singleton_array_is_rejected_without_scalar_conversion_warning() -> None:
    malformed = _nested_object_scalar(np.array([2]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="max_events_per_session"):
            event_limits._coerce_optional_nonnegative_integer(
                malformed,
                "max_events_per_session",
            )


def test_valid_nested_numeric_scalars_remain_supported() -> None:
    assert (
        event_limits._coerce_optional_nonnegative_integer(
            _nested_object_scalar(np.int64(3)),
            "max_events_per_session",
        )
        == 3
    )
    assert (
        event_limits._coerce_positive_integer(
            _nested_object_scalar(np.float64(2.0)),
            "n_cell_splits",
        )
        == 2
    )
    assert benchmarks._cell_split_seed(_nested_object_scalar(np.int64(7)), 3) == 10


def test_seed_validator_uses_active_nested_scalar_guard() -> None:
    assert (
        seed_validation._coerce_optional_nonnegative_integer
        is event_limits._coerce_optional_nonnegative_integer
    )
