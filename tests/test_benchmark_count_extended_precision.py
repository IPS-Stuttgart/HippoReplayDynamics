from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hipporeplayimm.benchmark_event_limit_validation import _coerce_optional_nonnegative_integer
from hipporeplayimm.benchmarks import BenchmarkConfig, _n_cell_splits


def _extended_precision_integer() -> tuple[np.longdouble, int]:
    expected = 2**53 + 1
    value = np.longdouble(str(expected))
    if int(value) != expected:
        pytest.skip("platform longdouble does not exceed binary64 integer precision")
    return value, expected


def test_event_limit_preserves_extended_precision_integer_exactly() -> None:
    value, expected = _extended_precision_integer()

    assert _coerce_optional_nonnegative_integer(value, "max_events_per_session") == expected


def test_cell_split_count_preserves_extended_precision_integer_exactly() -> None:
    value, expected = _extended_precision_integer()
    config = replace(BenchmarkConfig(), n_cell_splits=value)

    assert _n_cell_splits(config) == expected
