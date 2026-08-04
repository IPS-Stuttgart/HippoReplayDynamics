from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts.test_pf_within_event_replay_grammar import (
    build_gate_summary,
    build_grammar_decisions,
    cumulative_segment_scores,
    fixed_window_bounds,
    infer_local_grammar,
    summarize_grammar_test,
)


def test_fixed_window_bounds_merges_one_bin_remainder() -> None:
    assert fixed_window_bounds(11, 5) == [(0, 5), (5, 11)]
    assert fixed_window_bounds(12, 5) == [(0, 5), (5, 10), (10, 12)]
    assert fixed_window_bounds(1, 5) == []


def _local_scores(pattern: dict[str, list[float]], condition: str = "original", shuffle_index: int = -1) -> pd.DataFrame:
    rows = []
    n_windows = len(next(iter(pattern.values())))
    for window in range(n_windows):
        for mode, values in pattern.items():
            rows.append(
                {
                    "session": "Rat1/Open1",
                    "rat": "Rat1",
                    "event_index": 1,
                    "condition": condition,
                    "shuffle_index": shuffle_index,
                    "local_window_index": window,
                    "local_start_time_s": window * 0.02,
                    "local_end_time_s": (window + 1) * 0.02,
                    "local_duration_s": 0.02,
                    "mode": mode,
                    "local_log_evidence": values[window],
                    "event_n_spikes": 20,
                    "status": "success",
                }
            )
    return pd.DataFrame(rows)


def test_local_evidence_recovers_stationary_momentum_fragmented_grammar() -> None:
    local = _local_scores(
        {
            "stationary": [20, 20, -20, -20, -20, -20],
            "diffusion": [-20] * 6,
            "momentum": [-20, -20, 20, 20, -20, -20],
            "fragmented": [-20, -20, -20, -20, 20, 20],
        }
    )
    segments = cumulative_segment_scores(local)
    assert len(segments) == 4 * 21
    sequence, motifs = infer_local_grammar(local, max_segments=3, mode_switch_penalty=0.0)
    assert sequence["mode"].tolist() == ["stationary", "momentum", "fragmented"]
    assert bool(motifs.iloc[0]["ordered_trajectory_grammar"])


def test_matched_shuffle_summary_and_nonvacuous_gates() -> None:
    rows = []
    for rat_index in range(4):
        rat = f"Rat{rat_index + 1}"
        for event in range(2):
            rows.append(
                {
                    "session": f"{rat}/Open1",
                    "rat": rat,
                    "event_index": event,
                    "condition": "original",
                    "shuffle_index": -1,
                    "motif": "stationary->momentum",
                    "segment_count": 2,
                    "ordered_trajectory_grammar": True,
                }
            )
            for shuffle in range(20):
                rows.append(
                    {
                        "session": f"{rat}/Open1",
                        "rat": rat,
                        "event_index": event,
                        "condition": "shuffled",
                        "shuffle_index": shuffle,
                        "motif": "stationary",
                        "segment_count": 1,
                        "ordered_trajectory_grammar": False,
                    }
                )
    replicates = pd.DataFrame(rows)
    decisions = build_grammar_decisions(replicates)
    summary, by_rat = summarize_grammar_test(replicates, decisions)
    assert summary.iloc[0]["empirical_p_value_one_sided"] == 1 / 21
    assert (by_rat["ordered_trajectory_fraction_excess"] > 0).all()
    local = pd.DataFrame({"status": ["success"]})
    gates = build_gate_summary(local, replicates, summary, by_rat, expected_events=8, n_shuffles=20)
    assert bool(gates.set_index("gate").loc["overall", "passed"])

    empty_summary, empty_by_rat = summarize_grammar_test(pd.DataFrame(columns=replicates.columns), pd.DataFrame())
    empty_gates = build_gate_summary(pd.DataFrame(), pd.DataFrame(columns=replicates.columns), empty_summary, empty_by_rat, expected_events=0, n_shuffles=20)
    assert not bool(empty_gates.set_index("gate").loc["selected_events_present", "passed"])
