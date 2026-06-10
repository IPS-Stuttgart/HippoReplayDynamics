from __future__ import annotations

import pandas as pd

from scripts.triage_momentum_recovery import (
    build_momentum_recovery_triage,
    summarize_triage_events,
)


def _row(
    event: int,
    model: str,
    value: float,
    *,
    support: str = "exact_full_grid",
    comparable: bool = True,
    missing_bins: int = 0,
    coverage: float = 1.0,
) -> dict[str, object]:
    return {
        "matrix_id": "cfg-a",
        "session": "Rat1/Open1",
        "event_index": event,
        "simulation_event_index": event,
        "replicate": event,
        "true_model": "momentum",
        "expected_model": "sorted-spike-state-space-momentum",
        "model": model,
        "status": "success",
        "log_evidence": value,
        "evidence_support": support,
        "evidence_comparable": comparable,
        "candidate_true_bin_coverage": coverage,
        "candidate_true_pair_coverage": coverage,
        "candidate_true_triplet_coverage": coverage,
        "candidate_true_path_fully_supported": int(coverage >= 1.0 and missing_bins == 0),
        "candidate_true_path_missing_bins": missing_bins,
        "state_space_momentum_candidate_top_k": 128,
    }


def test_triage_separates_strict_certified_and_support_loss_events():
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-diffusion", 0.0),
            _row(0, "sorted-spike-state-space-momentum", 2.0),
            _row(1, "sorted-spike-state-space-diffusion", 1.0),
            _row(
                1,
                "sorted-spike-state-space-momentum",
                3.0,
                support="truncated_full_grid",
                comparable=False,
            ),
            _row(2, "sorted-spike-state-space-diffusion", 4.0),
            _row(
                2,
                "sorted-spike-state-space-momentum",
                2.0,
                support="truncated_full_grid",
                comparable=False,
                missing_bins=2,
                coverage=0.8,
            ),
        ]
    )

    tables = build_momentum_recovery_triage(scores)
    events = tables.event_table.set_index("event_index")

    assert events.loc[0, "triage_category"] == "strict_exact_recovery"
    assert events.loc[1, "triage_category"] == "lower_bound_certified_recovery"
    assert events.loc[2, "triage_category"] == "candidate_support_loss"
    assert bool(events.loc[1, "lower_bound_certified_recovery"])
    assert bool(events.loc[2, "candidate_support_loss"])

    summary = tables.summary.iloc[0]
    assert summary["momentum_events"] == 3
    assert summary["strict_exact_recovery_events"] == 1
    assert summary["lower_bound_certified_recovery_events"] == 1
    assert summary["candidate_support_loss_events"] == 1
    assert summary["certified_or_strict_recovery_events"] == 2

    assert len(tables.failure_examples) == 1
    assert tables.failure_examples.iloc[0]["triage_category"] == "candidate_support_loss"


def test_triage_marks_oracle_support_recovery_separately():
    scores = pd.DataFrame(
        [
            {
                **_row(0, "sorted-spike-state-space-diffusion", 0.0),
                "oracle_candidate_support": True,
            },
            {
                **_row(
                    0,
                    "sorted-spike-state-space-momentum",
                    5.0,
                    support="truncated_full_grid",
                    comparable=False,
                ),
                "oracle_candidate_support": True,
            },
        ]
    )

    events = build_momentum_recovery_triage(scores).event_table

    assert events.iloc[0]["triage_category"] == "oracle_support_recovers"
    assert bool(events.iloc[0]["certified_or_strict_recovery"])


def test_triage_parses_string_false_comparable_and_recovery_flags():
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-diffusion", 1.0),
            _row(
                0,
                "sorted-spike-state-space-momentum",
                3.0,
                support="truncated_full_grid",
                comparable="False",
            ),
        ]
    )

    tables = build_momentum_recovery_triage(scores)
    event = tables.event_table.iloc[0]

    assert event["triage_category"] == "lower_bound_certified_recovery"
    assert bool(event["expected_model_evidence_comparable"]) is False

    synthetic_events = pd.DataFrame(
        [
            {
                "triage_category": "exact_nonrecovery",
                "event_index": 0,
                "certified_or_strict_recovery": "False",
                "strict_exact_recovery": "False",
                "lower_bound_certified_recovery": "False",
                "candidate_support_loss": "False",
                "expected_minus_best_comparable_log_evidence": -1.0,
            }
        ]
    )
    summary = summarize_triage_events(synthetic_events).iloc[0]

    assert summary["certified_or_strict_recovery_events"] == 0
    assert summary["strict_exact_recovery_events"] == 0
