from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm.sweeps import (
    PyRecEstSweepConfig,
    aggregate_sweep_summary,
    pyrecest_parameter_grid,
)


def test_pyrecest_parameter_grid_rejects_fractional_random_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seed"):
        pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=1.5))


def test_pyrecest_parameter_grid_rejects_boolean_random_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="random_seed"):
        pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=True))


def test_pyrecest_parameter_grid_rejects_fractional_random_seed_sequence() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seeds"):
        pyrecest_parameter_grid(PyRecEstSweepConfig(random_seeds=(1, 2.5)))


def test_pyrecest_parameter_grid_rejects_textual_random_seed_sequence() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seeds"):
        pyrecest_parameter_grid(PyRecEstSweepConfig(random_seeds="12"))


def test_pyrecest_parameter_grid_canonicalizes_integer_like_random_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    rows = pyrecest_parameter_grid(PyRecEstSweepConfig(random_seed=np.array(2.0)))

    assert rows[0]["random_seed"] == 2
    assert isinstance(rows[0]["random_seed"], int)


def test_aggregate_sweep_summary_rejects_fractional_seed_aliasing() -> None:
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
