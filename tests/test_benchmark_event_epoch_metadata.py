from __future__ import annotations

import hipporeplayimm  # noqa: F401  # import applies runtime patches
from hipporeplayimm.benchmarks import BenchmarkConfig, _benchmark_config_metadata


def test_benchmark_metadata_records_event_epoch_scope() -> None:
    metadata = _benchmark_config_metadata(BenchmarkConfig(event_epoch="all"))

    assert metadata["benchmark_event_epoch"] == "all"


def test_benchmark_metadata_defaults_event_epoch_scope() -> None:
    metadata = _benchmark_config_metadata(BenchmarkConfig())

    assert metadata["benchmark_event_epoch"] == "run"
