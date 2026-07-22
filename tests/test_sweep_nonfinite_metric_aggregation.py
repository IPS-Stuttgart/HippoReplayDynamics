from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.sweeps import aggregate_sweep_summary


def test_aggregate_sweep_summary_ignores_nonfinite_metric_values() -> None:
    summary = pd.DataFrame(
        {
            "random_seed": [1, 2, 3],
            "pyrecest_model": ["pyrecest-goal-particle"] * 3,
            "goal_accuracy": [0.25, np.inf, -np.inf],
        }
    )

    aggregate = aggregate_sweep_summary(
        summary,
        group_columns=("pyrecest_model",),
        metric_columns=("goal_accuracy",),
    )
    row = aggregate.iloc[0]

    assert row["random_seed_count"] == 3
    assert row["goal_accuracy_n"] == 1
    assert row["goal_accuracy_mean"] == 0.25
    assert row["goal_accuracy_std"] == 0.0
    assert row["goal_accuracy_ci95_low"] == 0.25
    assert row["goal_accuracy_ci95_high"] == 0.25


def test_aggregate_sweep_summary_reports_empty_metric_when_all_values_are_nonfinite() -> None:
    summary = pd.DataFrame(
        {
            "random_seed": [1, 2],
            "pyrecest_model": ["pyrecest-goal-particle"] * 2,
            "goal_accuracy": [np.inf, -np.inf],
        }
    )

    aggregate = aggregate_sweep_summary(
        summary,
        group_columns=("pyrecest_model",),
        metric_columns=("goal_accuracy",),
    )
    row = aggregate.iloc[0]

    assert row["goal_accuracy_n"] == 0
    assert np.isnan(row["goal_accuracy_mean"])
    assert np.isnan(row["goal_accuracy_std"])
    assert np.isnan(row["goal_accuracy_ci95_low"])
    assert np.isnan(row["goal_accuracy_ci95_high"])
