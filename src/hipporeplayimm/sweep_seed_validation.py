"""Runtime validation for PyRecEst sweep random seeds and output artifacts."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

_PATCHED_FLAG = "_sweep_seed_validation_patch_applied"
_GRID_FLAG = "_sweep_seed_validation_parameter_grid_wrapper"
_BENCHMARK_FLAG = "_sweep_seed_validation_benchmark_config_wrapper"
_SORTED_FLAG = "_sweep_seed_validation_sorted_seed_wrapper"
_OUTPUT_FLAG = "_sweep_output_stale_cleanup_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_OPTIONAL_OUTPUT_FILENAMES = (
    "behavioral_ground_truth.csv",
    "ground_truth_comparison.csv",
    "pareto_summary.csv",
    "aggregate_summary.csv",
    "pareto_aggregate_summary.csv",
)


def apply_sweep_seed_validation_patch() -> None:
    """Install strict random-seed validation and output cleanup for sweep helpers."""

    from . import sweeps

    if getattr(sweeps, _PATCHED_FLAG, False) and _current(sweeps):
        _synchronize_output_writer_aliases(sweeps.write_pyrecest_sweep_outputs)
        return

    if not getattr(sweeps.pyrecest_parameter_grid, _GRID_FLAG, False):
        original_grid = sweeps.pyrecest_parameter_grid

        @wraps(original_grid)
        def pyrecest_parameter_grid(config):
            raw_seeds = (
                (getattr(config, "random_seed"),)
                if getattr(config, "random_seeds", None) is None
                else getattr(config, "random_seeds")
            )
            seed_name = "random_seed" if getattr(config, "random_seeds", None) is None else "random_seeds"
            seeds = _seed_sequence(raw_seeds, seed_name)
            validated = (
                replace(config, random_seed=seeds[0])
                if getattr(config, "random_seeds", None) is None
                else replace(config, random_seeds=seeds)
            )
            return original_grid(validated)

        _mark(pyrecest_parameter_grid, original_grid, _GRID_FLAG)
        sweeps.pyrecest_parameter_grid = pyrecest_parameter_grid

    if not getattr(sweeps._benchmark_config, _BENCHMARK_FLAG, False):
        original_benchmark = sweeps._benchmark_config

        @wraps(original_benchmark)
        def _benchmark_config(config, parameters: dict[str, object]):
            validated = dict(parameters)
            validated["random_seed"] = _seed_value(validated.get("random_seed"), "random_seed")
            return original_benchmark(config, validated)

        _mark(_benchmark_config, original_benchmark, _BENCHMARK_FLAG)
        sweeps._benchmark_config = _benchmark_config

    if not getattr(sweeps._sorted_numeric_values, _SORTED_FLAG, False):
        original_sorted = sweeps._sorted_numeric_values

        @wraps(original_sorted)
        def _sorted_numeric_values(values):
            return sorted({_seed_value(value, "random_seed") for value in values})

        _mark(_sorted_numeric_values, original_sorted, _SORTED_FLAG)
        sweeps._sorted_numeric_values = _sorted_numeric_values

    if not getattr(sweeps.write_pyrecest_sweep_outputs, _OUTPUT_FLAG, False):
        original_writer = sweeps.write_pyrecest_sweep_outputs

        @wraps(original_writer)
        def write_pyrecest_sweep_outputs(result, output):
            written = original_writer(result, output)
            present = _optional_output_presence(result, sweeps)
            output_path = Path(output)
            for filename in _OPTIONAL_OUTPUT_FILENAMES:
                if not present[filename]:
                    (output_path / filename).unlink(missing_ok=True)
            return written

        _mark(write_pyrecest_sweep_outputs, original_writer, _OUTPUT_FLAG)
        sweeps.write_pyrecest_sweep_outputs = write_pyrecest_sweep_outputs

    _synchronize_output_writer_aliases(sweeps.write_pyrecest_sweep_outputs)
    setattr(sweeps, _PATCHED_FLAG, True)


def _optional_output_presence(result: Any, sweeps: Any) -> dict[str, bool]:
    pareto = sweeps.pareto_sweep_summary(result.summary)
    aggregate = (
        sweeps.aggregate_sweep_summary(result.summary)
        if result.aggregate_summary.empty
        else result.aggregate_summary
    )
    pareto_aggregate = (
        sweeps.pareto_aggregate_sweep_summary(aggregate)
        if not aggregate.empty
        else aggregate
    )
    return {
        "behavioral_ground_truth.csv": result.behavioral_ground_truth is not None,
        "ground_truth_comparison.csv": not result.ground_truth_comparison.empty,
        "pareto_summary.csv": not pareto.empty,
        "aggregate_summary.csv": not aggregate.empty,
        "pareto_aggregate_summary.csv": not pareto_aggregate.empty,
    }


def _synchronize_output_writer_aliases(active: Any) -> None:
    """Refresh package modules that imported the sweep writer by value."""

    lineage: set[Any] = set()
    current = active
    while callable(current) and current not in lineage:
        lineage.add(current)
        current = getattr(current, _ORIGINAL_ATTR, None)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        alias = getattr(module, "write_pyrecest_sweep_outputs", None)
        if alias in lineage and alias is not active:
            setattr(module, "write_pyrecest_sweep_outputs", active)


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
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative integer") from exc
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a finite nonnegative integer")
    item: Any = raw.item()
    if isinstance(item, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not boolean")
    if isinstance(item, (int, np.integer)):
        seed = int(item)
    elif isinstance(item, (str, bytes)):
        seed = _seed_text_value(item, name)
    elif isinstance(item, Decimal):
        if not item.is_finite():
            raise ValueError(f"{name} must be a finite nonnegative integer")
        integer = item.to_integral_value()
        if item != integer:
            raise ValueError(f"{name} must be a finite nonnegative integer")
        seed = int(integer)
    else:
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite nonnegative integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a finite nonnegative integer")
        seed = int(numeric)
    if seed < 0:
        raise ValueError(f"{name} must be a finite nonnegative integer")
    return seed


def _seed_text_value(value: str | bytes, name: str) -> int:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} must be a finite nonnegative integer") from exc
    else:
        text = value
    text = text.strip()
    if not text:
        raise ValueError(f"{name} must be a finite nonnegative integer")
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        numeric = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite nonnegative integer") from exc
    if not numeric.is_finite():
        raise ValueError(f"{name} must be a finite nonnegative integer")
    integer = numeric.to_integral_value()
    if numeric != integer:
        raise ValueError(f"{name} must be a finite nonnegative integer")
    return int(integer)


def _current(sweeps) -> bool:
    return (
        getattr(sweeps.pyrecest_parameter_grid, _GRID_FLAG, False)
        and getattr(sweeps._benchmark_config, _BENCHMARK_FLAG, False)
        and getattr(sweeps._sorted_numeric_values, _SORTED_FLAG, False)
        and getattr(sweeps.write_pyrecest_sweep_outputs, _OUTPUT_FLAG, False)
    )


def _mark(function: Any, original: Any, flag: str) -> None:
    setattr(function, flag, True)
    setattr(function, _ORIGINAL_ATTR, original)


__all__ = ["apply_sweep_seed_validation_patch"]
