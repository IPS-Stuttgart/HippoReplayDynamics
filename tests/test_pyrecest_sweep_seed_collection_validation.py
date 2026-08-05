from __future__ import annotations

import pytest

from hipporeplayimm.sweeps import PyRecEstSweepConfig, pyrecest_parameter_grid


@pytest.mark.parametrize(
    "random_seeds",
    [
        {11, 12},
        frozenset({11, 12}),
        {11: "first", 12: "second"},
    ],
)
def test_pyrecest_parameter_grid_rejects_unordered_seed_collections(
    random_seeds: object,
) -> None:
    config = PyRecEstSweepConfig(random_seeds=random_seeds)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ordered iterable of random seeds"):
        pyrecest_parameter_grid(config)


def test_pyrecest_parameter_grid_preserves_ordered_seed_iterables() -> None:
    config = PyRecEstSweepConfig(random_seeds=[12, 11])  # type: ignore[arg-type]

    rows = pyrecest_parameter_grid(config)

    assert [row["random_seed"] for row in rows] == [12, 11]
