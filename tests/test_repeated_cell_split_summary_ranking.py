from __future__ import annotations

import pandas as pd

from scripts.repeated_cell_split_benchmark import _aggregate_summary


def test_aggregate_summary_ranks_by_leading_metric_without_index_column() -> None:
    rows = pd.DataFrame(
        {
            "model": ["stationary", "stationary", "momentum", "momentum"],
            "split_seed": [1, 2, 1, 2],
            "heldout_log_likelihood": [-10.0, -12.0, -8.0, -9.0],
        }
    )

    summary = _aggregate_summary(rows)

    assert "index" not in summary.columns
    assert summary["model"].tolist() == ["momentum", "stationary"]
    assert summary["heldout_log_likelihood_mean"].tolist() == [-8.5, -11.0]
