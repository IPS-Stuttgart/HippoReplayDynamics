import pandas as pd

from scripts.repeated_cell_split_benchmark import _aggregate_summary


def test_aggregate_summary_counts_split_seeds_per_model():
    rows = pd.DataFrame(
        {
            "model": ["stationary", "stationary", "momentum"],
            "split_seed": [1, 2, 1],
            "heldout_log_likelihood": [-10.0, -11.0, -9.0],
            "delta_vs_best_static": [0.0, 0.0, 2.0],
        }
    )

    summary = _aggregate_summary(rows)
    split_counts = dict(zip(summary["model"], summary["split_seeds"]))

    assert split_counts == {"stationary": 2, "momentum": 1}
