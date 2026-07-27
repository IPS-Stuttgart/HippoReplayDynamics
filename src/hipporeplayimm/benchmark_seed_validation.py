"""Validate benchmark random seeds before sampling or recording metadata.

Benchmark configuration reaches the runtime from Python, NumPy, notebooks, and
serialized tables. Raw ``int(...)`` coercion silently truncated fractional
seeds, accepted booleans as integers, and could make distinct requested runs
share one random stream. Canonicalize only exact nonnegative integer scalars
and reject malformed seed sequences before any benchmark work starts.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from functools import wraps
from typing import Any, Iterable

from .benchmark_event_limit_validation import _coerce_optional_nonnegative_integer

_PATCHED_FLAG = "_benchmark_seed_validation_patch_applied"
_RUN_FLAG = "_benchmark_seed_validation_run_wrapper"
_CELL_SPLIT_SEED_FLAG = "_benchmark_seed_validation_cell_split_seed_wrapper"
_EVENT_SUBSET_SEED_FLAG = "_benchmark_seed_validation_event_subset_seed_wrapper"
_SPLIT_CELLS_FLAG = "_benchmark_seed_validation_split_cells_wrapper"
_CONFIG_METADATA_FLAG = "_benchmark_seed_validation_config_metadata_wrapper"
_SPLIT_METADATA_FLAG = "_benchmark_seed_validation_split_metadata_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


class _BenchmarkSeedConfigProxy:
    """Delegate a config object while overriding canonical seed fields."""

    def __init__(self, config: object, overrides: dict[str, object]) -> None:
        self._config = config
        self._overrides = dict(overrides)

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._config, name)


def apply_benchmark_seed_validation_patch() -> None:
    """Install exact seed validation across benchmark entry points and helpers."""

    from . import benchmarks

    _patch_run_open_field_benchmark(benchmarks)
    _patch_cell_split_seed(benchmarks)
    _patch_event_subset_random_seed(benchmarks)
    _patch_split_cells(benchmarks)
    _patch_benchmark_config_metadata(benchmarks)
    _patch_benchmark_split_metadata(benchmarks)
    _synchronize_benchmark_runner_aliases(benchmarks.run_open_field_benchmark)
    setattr(benchmarks, _PATCHED_FLAG, True)


def _patch_run_open_field_benchmark(benchmarks: Any) -> None:
    current = benchmarks.run_open_field_benchmark
    if getattr(current, _RUN_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def run_open_field_benchmark(root, config=None):
        if config is None:
            return previous(root, config)
        validated = _config_with_validated_seeds(config, include_random_seeds=True)
        return previous(root, validated)

    _mark(run_open_field_benchmark, previous, _RUN_FLAG)
    benchmarks.run_open_field_benchmark = run_open_field_benchmark


def _patch_cell_split_seed(benchmarks: Any) -> None:
    current = benchmarks._cell_split_seed
    if getattr(current, _CELL_SPLIT_SEED_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def _cell_split_seed(base_seed, split_index):
        return previous(
            _seed_value(base_seed, "random_seed"),
            _seed_value(split_index, "split_index"),
        )

    _mark(_cell_split_seed, previous, _CELL_SPLIT_SEED_FLAG)
    benchmarks._cell_split_seed = _cell_split_seed


def _patch_event_subset_random_seed(benchmarks: Any) -> None:
    current = benchmarks._event_subset_random_seed
    if getattr(current, _EVENT_SUBSET_SEED_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def _event_subset_random_seed(config, split_index):
        validated = _config_with_validated_seeds(config, include_random_seeds=False)
        return previous(validated, _seed_value(split_index, "split_index"))

    _mark(_event_subset_random_seed, previous, _EVENT_SUBSET_SEED_FLAG)
    benchmarks._event_subset_random_seed = _event_subset_random_seed


def _patch_split_cells(benchmarks: Any) -> None:
    current = benchmarks._split_cells
    if getattr(current, _SPLIT_CELLS_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def _split_cells(cell_ids, test_fraction, random_seed):
        return previous(
            cell_ids,
            test_fraction,
            _seed_value(random_seed, "random_seed"),
        )

    _mark(_split_cells, previous, _SPLIT_CELLS_FLAG)
    benchmarks._split_cells = _split_cells


def _patch_benchmark_config_metadata(benchmarks: Any) -> None:
    current = benchmarks._benchmark_config_metadata
    if getattr(current, _CONFIG_METADATA_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def _benchmark_config_metadata(config):
        validated = _config_with_validated_seeds(config, include_random_seeds=False)
        return previous(validated)

    _mark(_benchmark_config_metadata, previous, _CONFIG_METADATA_FLAG)
    benchmarks._benchmark_config_metadata = _benchmark_config_metadata


def _patch_benchmark_split_metadata(benchmarks: Any) -> None:
    current = benchmarks._benchmark_split_metadata
    if getattr(current, _SPLIT_METADATA_FLAG, False):
        return

    previous = current

    @wraps(previous)
    def _benchmark_split_metadata(config, split_index):
        validated = _config_with_validated_seeds(config, include_random_seeds=False)
        return previous(validated, _seed_value(split_index, "split_index"))

    _mark(_benchmark_split_metadata, previous, _SPLIT_METADATA_FLAG)
    benchmarks._benchmark_split_metadata = _benchmark_split_metadata


def _config_with_validated_seeds(
    config: object,
    *,
    include_random_seeds: bool,
) -> object:
    overrides: dict[str, object] = {
        "random_seed": _seed_value(getattr(config, "random_seed", 1), "random_seed"),
    }
    event_subset_seed = getattr(config, "event_subset_seed", None)
    if event_subset_seed is not None:
        overrides["event_subset_seed"] = _seed_value(
            event_subset_seed,
            "event_subset_seed",
        )
    if include_random_seeds:
        random_seeds = getattr(config, "random_seeds", None)
        if random_seeds is not None:
            overrides["random_seeds"] = _seed_sequence(random_seeds, "random_seeds")

    try:
        return replace(config, **overrides)
    except TypeError:
        return _BenchmarkSeedConfigProxy(config, overrides)


def _seed_sequence(values: Iterable[object], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain at least one random seed")
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain at least one random seed") from exc
    if not raw_values:
        raise ValueError(f"{name} must contain at least one random seed")
    return tuple(_seed_value(value, name) for value in raw_values)


def _seed_value(value: object, name: str) -> int:
    candidate = _coerce_optional_nonnegative_integer(value, name)
    if candidate is None:
        raise ValueError(f"{name} must be a non-negative integer")
    return candidate


def _synchronize_benchmark_runner_aliases(active: Any) -> None:
    """Refresh package modules that imported the benchmark runner by value."""

    lineage: set[Any] = set()
    current = active
    while callable(current) and current not in lineage:
        lineage.add(current)
        current = getattr(current, _ORIGINAL_ATTR, None)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        alias = getattr(module, "run_open_field_benchmark", None)
        if callable(alias) and alias in lineage and alias is not active:
            setattr(module, "run_open_field_benchmark", active)


def _mark(function: Any, original: Any, flag: str) -> None:
    setattr(function, flag, True)
    setattr(function, _ORIGINAL_ATTR, original)


__all__ = [
    "apply_benchmark_seed_validation_patch",
    "_seed_sequence",
    "_seed_value",
]
