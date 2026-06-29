"""Restore benchmark scope metadata after metadata compatibility patching.

The score-table metadata compatibility layer replaces ``benchmarks._benchmark_config_metadata``
so post-hoc decoding can reconstruct older score tables.  Keep that replacement
synchronized with canonical benchmark scope metadata used by relative-metric
scoping.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def apply_benchmark_metadata_scope_patch() -> None:
    """Ensure benchmark-level split/event-subset metadata is emitted."""

    from . import benchmarks as bench

    metadata = bench._benchmark_config_metadata
    if getattr(metadata, "_benchmark_scope_metadata_wrapped", False):
        return

    def benchmark_config_metadata_with_scope_fields(config: Any) -> dict[str, object]:
        out = dict(metadata(config))
        out["benchmark_n_cell_splits"] = int(getattr(config, "n_cell_splits", 1))
        out["benchmark_randomize_event_subset"] = bool(
            getattr(config, "randomize_event_subset", False)
        )
        event_subset_seed = getattr(config, "event_subset_seed", None)
        out["benchmark_event_subset_base_seed"] = (
            np.nan if event_subset_seed is None else int(event_subset_seed)
        )
        out["benchmark_event_epoch"] = _event_epoch_scope(getattr(config, "event_epoch", "run"))
        return out

    benchmark_config_metadata_with_scope_fields._benchmark_scope_metadata_wrapped = True  # type: ignore[attr-defined]
    bench._benchmark_config_metadata = benchmark_config_metadata_with_scope_fields


def _event_epoch_scope(value: object) -> str:
    if value is None:
        return "run"
    label = str(value).strip()
    return label or "run"
