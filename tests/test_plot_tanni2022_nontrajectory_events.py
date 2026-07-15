from __future__ import annotations

import pandas as pd

from scripts.plot_tanni2022_nontrajectory_events import select_examples


def test_select_examples_uses_ambiguous_nontrajectory_best_rows_across_animals() -> None:
    decisions = pd.DataFrame(
        [
            {
                "animal": "A",
                "event_index": 1,
                "best_model": "stationary",
                "ordered_trajectory_confident": False,
                "delta_ordered_minus_static_or_fragmented": -2.0,
                "best_minus_runner_up_log_evidence": 2.0,
            },
            {
                "animal": "A",
                "event_index": 2,
                "best_model": "stationary",
                "ordered_trajectory_confident": False,
                "delta_ordered_minus_static_or_fragmented": -1.0,
                "best_minus_runner_up_log_evidence": 1.0,
            },
            {
                "animal": "B",
                "event_index": 3,
                "best_model": "stationary",
                "ordered_trajectory_confident": False,
                "delta_ordered_minus_static_or_fragmented": -1.5,
                "best_minus_runner_up_log_evidence": 1.5,
            },
            {
                "animal": "A",
                "event_index": 4,
                "best_model": "fragmented",
                "ordered_trajectory_confident": False,
                "delta_ordered_minus_static_or_fragmented": -3.0,
                "best_minus_runner_up_log_evidence": 3.0,
            },
            {
                "animal": "B",
                "event_index": 5,
                "best_model": "fragmented",
                "ordered_trajectory_confident": False,
                "delta_ordered_minus_static_or_fragmented": -2.5,
                "best_minus_runner_up_log_evidence": 2.5,
            },
            {
                "animal": "C",
                "event_index": 6,
                "best_model": "fragmented",
                "ordered_trajectory_confident": True,
                "delta_ordered_minus_static_or_fragmented": 6.0,
                "best_minus_runner_up_log_evidence": 6.0,
            },
        ]
    )

    selected = select_examples(decisions, examples_per_group=2)

    assert set(selected["event_index"]) == {1, 3, 4, 5}
    assert selected.groupby("diagnostic_group").size().to_dict() == {
        "fragmented_best_ambiguous": 2,
        "stationary_best_ambiguous": 2,
    }
    assert not selected["ordered_trajectory_confident"].any()
    assert selected.groupby("diagnostic_group")["animal"].nunique().eq(2).all()
