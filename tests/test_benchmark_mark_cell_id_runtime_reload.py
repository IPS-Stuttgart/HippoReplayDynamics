from __future__ import annotations

import importlib

import hipporeplayimm
import hipporeplayimm.benchmarks as benchmarks
import hipporeplayimm.ground_truth as ground_truth
from hipporeplayimm.benchmark_mark_cell_id_validation import _PATCHED_FLAG, _WRAPPER_FLAG


def test_benchmark_mark_cell_id_patch_restores_after_benchmarks_reload() -> None:
    hipporeplayimm.apply_runtime_patches()
    assert getattr(benchmarks, _PATCHED_FLAG, False)
    assert getattr(benchmarks._session_with_mark_cell_subset, _WRAPPER_FLAG, False)

    importlib.reload(benchmarks)

    # importlib.reload() reuses the module dictionary, so dynamic sentinel
    # attributes survive even though source-defined functions are replaced.
    assert getattr(benchmarks, _PATCHED_FLAG, False)
    assert not getattr(benchmarks._session_with_mark_cell_subset, _WRAPPER_FLAG, False)

    hipporeplayimm.apply_runtime_patches()

    assert getattr(benchmarks._session_with_mark_cell_subset, _WRAPPER_FLAG, False)
    assert ground_truth._session_with_mark_cell_subset is benchmarks._session_with_mark_cell_subset
