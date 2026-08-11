from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
import pytest

from hipporeplayimm.benchmarks import BenchmarkConfig
from hipporeplayimm.clusterless import ClusterlessMarkConfig
from hipporeplayimm.clusterless_config_validation import (
    _validate_benchmark_clusterless_mark_config,
    _validate_clusterless_mark_config,
)


def _object_scalar(value: object) -> np.ndarray:
    wrapper = np.empty((), dtype=object)
    wrapper[()] = value
    return wrapper


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mark_prior_count", "1.0", "mark_prior_count"),
        ("mark_prior_count", _object_scalar(np.asarray(True)), "mark_prior_count"),
        ("mark_kde_bandwidth", _object_scalar(np.asarray("2.0")), "mark_kde_bandwidth"),
        ("mark_kde_max_neighbors", _object_scalar(np.asarray([32])), "mark_kde_max_neighbors"),
    ],
)
def test_clusterless_config_rejects_malformed_scalar_wrappers(
    field: str,
    value: object,
    message: str,
) -> None:
    config = ClusterlessMarkConfig(**{field: value})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=message):
            _validate_clusterless_mark_config(config)


def test_benchmark_clusterless_config_rejects_nested_boolean_scalar() -> None:
    config = replace(
        BenchmarkConfig(),
        clusterless_mark_prior_count=_object_scalar(np.asarray(True)),
    )

    with pytest.raises(ValueError, match="mark_prior_count"):
        _validate_benchmark_clusterless_mark_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mark_prior_count", _object_scalar(np.asarray(np.float64(2.5)))),
        ("mark_kde_max_neighbors", _object_scalar(np.asarray(np.int64(32)))),
    ],
)
def test_clusterless_config_preserves_nested_real_scalars(field: str, value: object) -> None:
    config = ClusterlessMarkConfig(**{field: value})
    _validate_clusterless_mark_config(config)


def test_clusterless_config_rejects_cyclic_zero_dimensional_wrapper() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    config = ClusterlessMarkConfig(mark_prior_count=cyclic)

    with pytest.raises(ValueError, match="mark_prior_count"):
        _validate_clusterless_mark_config(config)
