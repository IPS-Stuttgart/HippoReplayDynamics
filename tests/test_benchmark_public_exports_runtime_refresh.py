from __future__ import annotations

import importlib

import hipporeplayimm
import hipporeplayimm.benchmarks as benchmarks


def test_runtime_patches_refresh_public_benchmark_exports_after_reload() -> None:
    hipporeplayimm.apply_runtime_patches()
    reloaded = importlib.reload(benchmarks)
    try:
        # Reload recreates source-defined benchmark classes and the entry point,
        # while package-level aliases still point at the pre-reload objects.
        assert hipporeplayimm.BenchmarkConfig is not reloaded.BenchmarkConfig
        assert hipporeplayimm.BenchmarkResult is not reloaded.BenchmarkResult
        assert hipporeplayimm.run_open_field_benchmark is not reloaded.run_open_field_benchmark

        hipporeplayimm.apply_runtime_patches()

        assert hipporeplayimm.BenchmarkConfig is reloaded.BenchmarkConfig
        assert hipporeplayimm.BenchmarkResult is reloaded.BenchmarkResult
        assert hipporeplayimm.run_open_field_benchmark is reloaded.run_open_field_benchmark
    finally:
        hipporeplayimm.apply_runtime_patches()
