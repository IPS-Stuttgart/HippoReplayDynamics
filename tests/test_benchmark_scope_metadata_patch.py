from __future__ import annotations

import numpy as np


def test_runtime_metadata_patch_preserves_scope_columns() -> None:
    import hipporeplayimm
    from hipporeplayimm import benchmarks

    hipporeplayimm.apply_runtime_patches()

    config = benchmarks.BenchmarkConfig(
        n_cell_splits=3,
        randomize_event_subset=True,
        event_subset_seed=29,
    )
    metadata = benchmarks._benchmark_config_metadata(config)

    assert metadata["benchmark_n_cell_splits"] == 3
    assert metadata["benchmark_randomize_event_subset"] is True
    assert metadata["benchmark_event_subset_base_seed"] == 29


def test_runtime_metadata_patch_uses_nan_for_missing_event_subset_seed() -> None:
    import hipporeplayimm
    from hipporeplayimm import benchmarks

    hipporeplayimm.apply_runtime_patches()

    metadata = benchmarks._benchmark_config_metadata(
        benchmarks.BenchmarkConfig(randomize_event_subset=False, event_subset_seed=None)
    )

    assert metadata["benchmark_randomize_event_subset"] is False
    assert np.isnan(metadata["benchmark_event_subset_base_seed"])
