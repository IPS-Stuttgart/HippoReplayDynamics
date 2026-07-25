from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.sweeps import (
    pareto_aggregate_sweep_summary,
    pareto_sweep_summary,
)


def test_pareto_sweep_summary_excludes_rows_with_missing_objectives() -> None:
    summary = pd.DataFrame(
        {
            "sweep_id": [0, 1],
            "goal_accuracy": [1.0, 0.9],
            "mean_delta_vs_best_static": [np.nan, 0.5],
        }
    )

    pareto = pareto_sweep_summary(summary)

    assert list(pareto["sweep_id"]) == [1]


def test_pareto_sweep_summary_excludes_infinite_objectives() -> None:
    summary = pd.DataFrame(
        {
            "sweep_id": [0, 1],
            "goal_accuracy": [np.inf, 0.9],
            "mean_delta_vs_best_static": [1.0, 0.5],
        }
    )

    pareto = pareto_sweep_summary(summary)

    assert list(pareto["sweep_id"]) == [1]


def test_pareto_aggregate_summary_requires_complete_mean_objectives() -> None:
    aggregate = pd.DataFrame(
        {
            "sweep_id": [0, 1],
            "goal_accuracy_mean": [1.0, 0.9],
            "mean_delta_vs_best_static_mean": [np.nan, 0.5],
        }
    )

    pareto = pareto_aggregate_sweep_summary(aggregate)

    assert list(pareto["sweep_id"]) == [1]
