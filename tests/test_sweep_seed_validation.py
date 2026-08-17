from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm.sweeps import PyRecEstSweepConfig, aggregate_sweep_summary, pyrecest_parameter_grid


def _nested_object_scalar(value: object, *, depth: int = 1) -> np.ndarray:
    current = value
    for _ in range(depth):
        wrapper = np.empty((), dtype=object)
        wrapper[()] = current
        current = wrapper
    return current


def test_pyrecest_parameter_grid_rejects_fractional_random_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seed"):
        pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=1.5))


def test_pyrecest_parameter_grid_canonicalizes_integer_like_random_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    rows = pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=np.array(2.0)))

    assert rows[0]["random_seed"] == 2
    assert isinstance(rows[0]["random_seed"], int)


@pytest.mark.parametrize(
    "seed",
    [
        _nested_object_scalar(True, depth=2),
        _nested_object_scalar(np.bool_(False), depth=2),
    ],
)
def test_pyrecest_parameter_grid_rejects_nested_array_wrapped_boolean_seed(seed: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="random_seed"):
        pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=seed))


@pytest.mark.parametrize(
    "seed",
    [
        _nested_object_scalar(np.array([1])),
        _nested_object_scalar(np.array([True])),
        _nested_object_scalar(np.array([[1]])),
    ],
)
def test_pyrecest_parameter_grid_rejects_nested_non_scalar_seed(seed: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seed"):
        pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=seed))


def test_pyrecest_parameter_grid_accepts_nested_zero_dimensional_numeric_seed() -> None:
    hipporeplayimm.apply_runtime_patches()
    seed = _nested_object_scalar(np.int64(7), depth=3)

    rows = pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=seed))

    assert rows[0]["random_seed"] == 7
    assert isinstance(rows[0]["random_seed"], int)


def test_aggregate_sweep_summary_rejects_fractional_seed() -> None:
    hipporeplayimm.apply_runtime_patches()
    summary = pd.DataFrame(
        {
            "random_seed": [1.5],
            "pyrecest_model": ["pyrecest-goal-particle"],
            "pyrecest_particles": [64],
            "goal_accuracy": [0.25],
        }
    )

    with pytest.raises(ValueError, match="random_seed"):
        aggregate_sweep_summary(summary)
