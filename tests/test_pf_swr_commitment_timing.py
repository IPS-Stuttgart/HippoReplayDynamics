from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.test_pf_swr_commitment_timing import (
    build_gates,
    deduplicate_physical_candidate_windows,
    next_local_departure,
    summarize,
)


def test_physical_window_dedup_collapses_overlaps_across_source_events() -> None:
    decisions = pd.DataFrame(
        [
            {"selection_rule": "strongest_exact_margin", "session": "Rat1/Open1", "rat": "Rat1", "event_index": 1, "null_index": 0, "window_start_s": 10.0, "window_end_s": 10.2, "trajectory_minus_nontrajectory_log_evidence": 8.0},
            {"selection_rule": "strongest_exact_margin", "session": "Rat1/Open1", "rat": "Rat1", "event_index": 2, "null_index": 0, "window_start_s": 10.1, "window_end_s": 10.3, "trajectory_minus_nontrajectory_log_evidence": 12.0},
            {"selection_rule": "strongest_exact_margin", "session": "Rat1/Open1", "rat": "Rat1", "event_index": 3, "null_index": 0, "window_start_s": 20.0, "window_end_s": 20.2, "trajectory_minus_nontrajectory_log_evidence": 9.0},
        ]
    )
    selected = deduplicate_physical_candidate_windows(decisions)
    assert len(selected) == 2
    assert selected.iloc[0]["event_index"] == 2
    assert selected.iloc[0]["physical_candidate_cluster_size"] == 2


def test_next_local_departure_requires_sustained_speed() -> None:
    times = np.arange(0.0, 2.0, 0.1)
    speed = np.zeros_like(times)
    speed[8] = 6.0
    speed[12:] = 6.0
    trace = np.column_stack([times, np.zeros_like(times), np.zeros_like(times), speed])
    result = next_local_departure(
        trace,
        0.5,
        speed_threshold_cm_s=5.0,
        minimum_departure_s=0.25,
    )
    assert result["local_pause_status"] == "local_pause"
    assert np.isclose(result["time_to_local_departure_s"], 0.7)


def test_summary_stays_insufficient_below_ten_physical_candidates() -> None:
    rows = []
    for rat in ("Rat1", "Rat2", "Rat3"):
        for event_class, values in (("off_swr", [4.0, 5.0]), ("swr", [1.0, 2.0, 3.0])):
            for index, value in enumerate(values):
                rows.append(
                    {
                        "event_class": event_class,
                        "session": f"{rat}/Open1",
                        "rat": rat,
                        "event_index": index,
                        "local_pause_status": "local_pause",
                        "time_to_local_departure_s": value,
                    }
                )
    events = pd.DataFrame(rows)
    summary, by_rat, _ = summarize(
        events,
        permutation_replicates=99,
        bootstrap_replicates=99,
        seed=1,
    )
    assert summary.iloc[0]["physical_off_swr_events"] == 6
    assert summary.iloc[0]["inferential_status"] == "insufficient"
    assert (by_rat["positive_direction"]).all()
    gates = build_gates(events, summary)
    assert not bool(gates.set_index("gate").loc["minimum_independent_off_swr_events", "passed"])
    assert not bool(gates.set_index("gate").loc["overall_inferential", "passed"])
