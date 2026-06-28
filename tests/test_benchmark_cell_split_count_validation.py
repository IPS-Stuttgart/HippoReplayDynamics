from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hipporeplayimm.benchmarks import (
    BenchmarkConfig,
    _benchmark_config_metadata,
    _benchmark_split_metadata,
    _n_cell_splits,
)


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_n_cell_splits_rejects_boolean_values(value: object) -> None:
    config = replace(BenchmarkConfig(), n_cell_splits=value)

    with pytest.raises(ValueError, match="n_cell_splits must be a positive integer"):
        _n_cell_splits(config)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        1.5,
        "1.5",
        float("nan"),
        np.array([1, 2]),
    ],
)
def test_n_cell_splits_rejects_nonpositive_or_noninteger_values(value: object) -> None:
    config = replace(BenchmarkConfig(), n_cell_splits=value)

    with pytest.raises(ValueError, match="n_cell_splits must be a positive integer"):
        _n_cell_splits(config)


def test_n_cell_splits_accepts_integer_like_metadata_values() -> None:
    config = replace(BenchmarkConfig(), n_cell_splits="2")

    assert _n_cell_splits(config) == 2
    assert _benchmark_config_metadata(config)["benchmark_n_cell_splits"] == 2
    assert _benchmark_split_metadata(config, 1)["benchmark_cell_split_count"] == 2


def test_n_cell_splits_metadata_rejects_boolean_values() -> None:
    config = replace(BenchmarkConfig(), n_cell_splits=True)

    with pytest.raises(ValueError, match="n_cell_splits must be a positive integer"):
        _benchmark_config_metadata(config)
    with pytest.raises(ValueError, match="n_cell_splits must be a positive integer"):
        _benchmark_split_metadata(config, 0)
