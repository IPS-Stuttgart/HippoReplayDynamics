from __future__ import annotations

import pandas as pd

from hipporeplayimm.sign_flip_report import score_table_sign_flip_summary


def test_monte_carlo_summary_is_independent_of_model_group_order() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    forward = pd.DataFrame(
        {
            "model": ["alpha"] * len(values) + ["beta"] * len(values),
            "delta_vs_best_static": values * 2,
        }
    )
    reverse = pd.DataFrame(
        {
            "model": ["beta"] * len(values) + ["alpha"] * len(values),
            "delta_vs_best_static": values * 2,
        }
    )
    kwargs = {
        "max_exact_n": 0,
        "n_permutations": 101,
        "random_seed": 11,
        "chunk_size": 17,
    }

    forward_summary = (
        score_table_sign_flip_summary(forward, **kwargs)
        .sort_values("model")
        .reset_index(drop=True)
    )
    reverse_summary = (
        score_table_sign_flip_summary(reverse, **kwargs)
        .sort_values("model")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(forward_summary, reverse_summary)
    assert forward_summary["random_seed"].tolist() == [11, 12]
