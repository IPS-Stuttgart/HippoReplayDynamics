from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.benchmarks import BenchmarkConfig
from hipporeplayimm.clusterless import ClusterlessMarkConfig
from hipporeplayimm.clusterless_config_validation import (
    _validate_benchmark_clusterless_mark_config,
    _validate_clusterless_mark_config,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mark_smoothing_sigma_bins", "1.0", "mark_smoothing_sigma_bins"),
        ("mark_prior_count", np.str_("1.0"), "mark_prior_count"),
        ("mark_variance_floor", "1.0", "mark_variance_floor"),
        ("rate_floor_hz", "0.001", "rate_floor_hz"),
        ("mark_kde_bandwidth", "2.0", "mark_kde_bandwidth"),
        ("mark_kde_spatial_sigma_bins", "1.0", "mark_kde_spatial_sigma_bins"),
        ("mark_kde_max_neighbors", "32", "mark_kde_max_neighbors"),
    ],
)
def test_clusterless_mark_config_rejects_numeric_string_scalars(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_clusterless_mark_config(ClusterlessMarkConfig(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clusterless_mark_smoothing_sigma_bins", "1.0", "mark_smoothing_sigma_bins"),
        ("clusterless_mark_prior_count", "1.0", "mark_prior_count"),
        ("clusterless_mark_variance_floor", "1.0", "mark_variance_floor"),
        ("clusterless_rate_floor_hz", "0.001", "rate_floor_hz"),
        ("clusterless_mark_kde_bandwidth", "2.0", "mark_kde_bandwidth"),
        ("clusterless_mark_kde_spatial_sigma_bins", "1.0", "mark_kde_spatial_sigma_bins"),
        ("clusterless_mark_kde_max_neighbors", np.asarray("32"), "mark_kde_max_neighbors"),
    ],
)
def test_benchmark_clusterless_mark_config_rejects_numeric_string_scalars(field: str, value: object, message: str) -> None:
    config = BenchmarkConfig()
    setattr(config, field, value)

    with pytest.raises(ValueError, match=message):
        _validate_benchmark_clusterless_mark_config(config)
