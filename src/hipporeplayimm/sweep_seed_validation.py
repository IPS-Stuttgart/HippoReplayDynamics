"""Validate PyRecEst sweep random-seed fields before integer coercion.

Sweep rows are later passed into benchmark construction with ``int(...)`` casts and
aggregate summaries stringify the seed set after converting through pandas.  Without
an explicit guard, corrupted values such as ``True`` or ``1.5`` can alias to seed
``1`` and mix distinct stochastic replicates.  This patch keeps integer-like
MATLAB/CSV values such as ``2.0`` valid while rejecting booleans, arrays, NaN, and
fractional seeds.
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any, Iterable

import numpy as np

_PATCHED_FLAG = "_sweep_seed_validation_patch_applied"
_PARAMETER_GRID_WRAPPER_FLAG = "_sweep_seed_validation_parameter_grid_wrapper"
_BENCHMARK_CONFIG_WRAPPER_FLAG = "_sweep_seed_validation_benchmark_config_wrapper"
_SORTED_SEEDS_WRAPPER_FLAG = "_sweep_seed_validation_sorted_seed_wrapper"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def apply_sweep_seed_validation_patch() -> None:
    """Install strict random-seed validation for PyRecEst sweep helpers."""

    from . import sweeps

    if getattr(sweeps, _PATCHED_FLAG, False) and _sweep_seed_wrappers_are_current(sweeps):
        return

    if not getattr(sweeps.pyrecest_parameter_grid, _PARAMETER_GRID_WRAPPER_FLAG, False):
        original_parameter_grid = sweeps.pyrecest_parameter_grid

        @wraps(original_parameter_grid)
        def pyrecest_parameter_grid(config):
            seeds = _validated_seed_sequence(
                (getattr(config, "random_seed"),)
                if getattr(config, "random_seeds", None) is None
                else getattr(config, "random_seeds"),
                "random_seed" if getattr(config, "random_seeds", None) is None else "random_seeds",
            )
            if getattr(config, "random_seeds", None) is None:
                validated_config = replace(config, random_seed=seeds[0])
            else:
                validated_config = replace(config, random_seeds=seeds)
            return original_parameter_grid(validated_config)

        _mark_wrapper(pyrecest_parameter_grid, original_parameter_grid, _PARAMETER_GRID_WRAPPER_FLAG)
        sweeps.pyrecest_parameter_grid = pyrecest_parameter_grid

    if not getattr(sweeps._benchmark_config, _BENCHMARK_CONFIG_WRAPPER_FLAG, False):
        original_benchmark_config = sweeps._benchmark_config

        @wraps(original_benchmark_config)
        def _benchmark_config(config, parameters: dict[str, object]):
            validated = dict(parameters)
            validated["random_seed"] = _nonnegative_integer_seed(
                validated.get("random_seed"),
                "random_seed",
            )
            return original_benchmark_config(config, validated)

        _mark_wrapper(_benchmark_config, original_benchmark_config, _BENCHMARK_CONFIG_WRAPPER_FLAG)
        sweeps._benchmark_config = _benchmark_config

    if not getattr(sweeps._sorted_numeric_values, _SORTED_SEEDS_WRAPPER_FLAG, False):
        original_sorted_numeric_values = sweeps._sorted_numeric_values

        @wraps(original_sorted_numeric_values)
        def _sorted_numeric_values(values):
            return sorted(
                set(
                    _nonnegative_integer_seed(value, "random_seed")
                    for value in values
                )
            )

        _mark_wrapper(_sorted_numeric_values, original_sorted_numeric_values, _SORTED_SEEDS_WRAPPER_FLAG)
        sweeps._sorted_numeric_values = _sorted_numeric_values

    setattr(sweeps, _PATCHED_FLAG, True)


def _validated_seed_sequence(values: Iterable[object], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain at least one random seed")
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain at least one random seed") from exc
    if not raw_values:
        raise ValueError(f"{name} must contain at least one random seed")
    return tuple(_nonnegative_integer_seed(value, name) for value in raw_values)


def _nonnegative_integer_seed(value: object, name: str) -> int:
    """Return a nonnegative integer seed without boolean or fractional aliasing."""

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


def _sweep_seed_wrappers_are_current(sweeps) -> bool:
    return (
        getattr(sweeps.pyrecest_parameter_grid, _PARAMETER_GRID_WRAPPER_FLAG, False)
        and getattr(sweeps._benchmark_config, _BENCHMARK_CONFIG_WRAPPER_FLAG, False)
        and getattr(sweeps._sorted_numeric_values, _SORTED_SEEDS_WRAPPER_FLAG, False)
    )


def _mark_wrapper(function: Any, original: Any, flag: str) -> None:
    setattr(function, flag, True)
    setattr(function, _ORIGINAL_ATTR, original)


__all__ = ["apply_sweep_seed_validation_patch"]
