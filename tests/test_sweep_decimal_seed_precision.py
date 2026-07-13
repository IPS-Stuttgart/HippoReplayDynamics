from __future__ import annotations

from decimal import Decimal

import pandas as pd

from hipporeplayimm.sweeps import aggregate_sweep_summary


def test_sweep_aggregation_preserves_exact_decimal_seed_identity() -> None:
    first = 2**53
    second = first + 1
    summary = pd.DataFrame(
        {
            "random_seed": pd.Series(
                [Decimal(str(first)), Decimal(str(second))],
                dtype=object,
            ),
            "pyrecest_model": ["pyrecest-goal-particle"] * 2,
            "pyrecest_particles": [128] * 2,
            "events": [1, 1],
        }
    )

    aggregate = aggregate_sweep_summary(summary)

    assert aggregate.loc[0, "random_seed_count"] == 2
    assert aggregate.loc[0, "random_seeds"] == f"{first},{second}"
