from __future__ import annotations

import pandas as pd

from scripts.audit_second_order_lower_bound_gap import (
    build_lower_bound_gap_tables,
    summarize_gap_table,
)


def _row(event: int, model: str, value: float, support: str, top_k: int, *, status: object = "success") -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": event,
        "model": model,
        "log_evidence": value,
        "evidence_support": support,
        "status": status,
        "state_space_diffusion_sigma_cm_sqrt_s": 85.0,
        "state_space_momentum_sigma_cm_sqrt_s": 85.0,
        "state_space_momentum_initial_sigma_cm_sqrt_s": 85.0,
        "state_space_momentum_velocity_decay": 0.95,
        "state_space_momentum_candidate_top_k": top_k,
        "state_space_momentum_predicted_candidate_top_k": 8,
        "state_space_momentum_candidate_source": "emission",
        "min_candidate_log_mass": -0.5,
    }


def test_lower_bound_gap_pairs_truncated_rows_to_exact_rows():
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-momentum", 10.0, "exact_full_grid", 0),
            _row(0, "sorted-spike-state-space-momentum", 8.0, "truncated_full_grid", 128),
            _row(1, "sorted-spike-state-space-momentum", 5.0, "exact_full_grid", 0),
            _row(1, "sorted-spike-state-space-momentum", 5.5, "truncated_full_grid", 128),
            _row(0, "sorted-spike-state-space-diffusion", 9.0, "exact_full_grid", 0),
        ]
    )

    tables = build_lower_bound_gap_tables(scores)
    gaps = tables.event_gaps.sort_values("event_index")

    assert len(gaps) == 2
    assert gaps.iloc[0]["lower_bound_gap_log_evidence"] == 2.0
    assert gaps.iloc[1]["lower_bound_gap_log_evidence"] == -0.5
    assert not bool(gaps.iloc[0]["lower_bound_exceeds_exact"])
    assert bool(gaps.iloc[1]["lower_bound_exceeds_exact"])

    summary = tables.summary.iloc[0]
    assert summary["paired_event_rows"] == 2
    assert summary["events_with_negative_gap"] == 1
    assert summary["mean_lower_bound_gap_log_evidence"] == 0.75


def test_lower_bound_gap_ignores_non_success_score_rows():
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-momentum", 100.0, "exact_full_grid", 0, status="failed"),
            _row(0, "sorted-spike-state-space-momentum", 8.0, "truncated_full_grid", 128),
            _row(1, "sorted-spike-state-space-momentum", 10.0, "exact_full_grid", 0),
            _row(1, "sorted-spike-state-space-momentum", 9.0, "truncated_full_grid", 128),
        ]
    )

    tables = build_lower_bound_gap_tables(scores)

    assert list(tables.event_gaps["event_index"]) == [1]
    assert tables.event_gaps.iloc[0]["lower_bound_gap_log_evidence"] == 1.0


def test_lower_bound_gap_treats_blank_status_as_legacy_success():
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-momentum", 10.0, "exact_full_grid", 0, status=""),
            _row(0, "sorted-spike-state-space-momentum", 8.0, "truncated_full_grid", 128, status=pd.NA),
            _row(1, "sorted-spike-state-space-momentum", 100.0, "exact_full_grid", 0, status="failed"),
            _row(1, "sorted-spike-state-space-momentum", 9.0, "truncated_full_grid", 128, status="nan"),
        ]
    )

    tables = build_lower_bound_gap_tables(scores)

    assert list(tables.event_gaps["event_index"]) == [0]
    assert tables.event_gaps.iloc[0]["lower_bound_gap_log_evidence"] == 2.0


def test_lower_bound_gap_returns_empty_without_exact_pairs():
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-momentum", 8.0, "truncated_full_grid", 128),
            _row(1, "sorted-spike-state-space-momentum", 7.0, "truncated_full_grid", 128),
        ]
    )

    tables = build_lower_bound_gap_tables(scores)

    assert tables.event_gaps.empty
    assert tables.summary.empty


def test_lower_bound_gap_summary_parses_string_bool_flags():
    event_gaps = pd.DataFrame(
        [
            {
                "model": "sorted-spike-state-space-momentum",
                "lower_bound_gap_log_evidence": 2.0,
                "lower_bound_exceeds_exact": "False",
                "lower_bound_gap_within_1": "0.0",
                "lower_bound_gap_within_3": "True",
                "lower_bound_gap_within_10": "True",
            },
            {
                "model": "sorted-spike-state-space-momentum",
                "lower_bound_gap_log_evidence": -0.5,
                "lower_bound_exceeds_exact": 1.0,
                "lower_bound_gap_within_1": "1.0",
                "lower_bound_gap_within_3": "True",
                "lower_bound_gap_within_10": "True",
            },
        ]
    )

    summary = summarize_gap_table(event_gaps).iloc[0]

    assert summary["events_with_negative_gap"] == 1
    assert summary["gap_within_1_fraction"] == 0.5
    assert summary["gap_within_3_fraction"] == 1.0
