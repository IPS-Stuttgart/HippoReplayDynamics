from __future__ import annotations

import pandas as pd

from hipporeplayimm.evidence_reporting import simulation_event_best_rows
from hipporeplayimm.simulation_recovery import (
    add_evidence_columns,
    certified_vs_exact_event_recovery,
    certified_vs_exact_recovery_summary,
    expected_scoring_model,
    recovery_summary,
)


def _row(
    event_id: str,
    true_model: str,
    model: str,
    log_evidence: float,
    **extra: object,
) -> dict[str, object]:
    row = {
        "status": "success",
        "session": "RatX/OpenY",
        "simulation_random_seed": 1,
        "simulation_event_index": 0,
        "event_index": 0,
        "event_id": event_id,
        "true_model": true_model,
        "expected_model": expected_scoring_model(true_model),
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
    }
    row.update(extra)
    return row


def test_simulation_recovery_keeps_same_event_index_separate_by_event_id() -> None:
    rows = pd.DataFrame(
        [
            _row("evt-a", "stationary", "sorted-spike-state-space-stationary", 0.0),
            _row("evt-a", "stationary", "sorted-spike-state-space-diffusion", -4.0),
            _row("evt-b", "stationary", "sorted-spike-state-space-stationary", -4.0),
            _row("evt-b", "stationary", "sorted-spike-state-space-diffusion", 0.0),
        ]
    )

    scored = add_evidence_columns(rows)
    best = simulation_event_best_rows(scored).sort_values("event_id").reset_index(drop=True)
    summary = recovery_summary(scored)
    overall = summary[summary["true_model"] == "overall"].iloc[0]

    assert best["event_id"].tolist() == ["evt-a", "evt-b"]
    assert best["model"].tolist() == [
        "sorted-spike-state-space-stationary",
        "sorted-spike-state-space-diffusion",
    ]
    assert overall["simulated_events"] == 2
    assert overall["recovered_events"] == 1


def test_certified_recovery_keeps_same_event_index_separate_by_event_id() -> None:
    rows = pd.DataFrame(
        [
            _row("evt-a", "momentum", "sorted-spike-state-space-diffusion", -1.0),
            _row(
                "evt-a",
                "momentum",
                "sorted-spike-state-space-momentum",
                2.0,
                diagnostic_state_space_momentum_evidence_support="truncated_full_grid",
            ),
            _row("evt-b", "momentum", "sorted-spike-state-space-diffusion", 5.0),
            _row(
                "evt-b",
                "momentum",
                "sorted-spike-state-space-momentum",
                1.0,
                diagnostic_state_space_momentum_evidence_support="truncated_full_grid",
            ),
        ]
    )

    scored = add_evidence_columns(rows)
    events = certified_vs_exact_event_recovery(scored).sort_values("event_id").reset_index(drop=True)
    summary = certified_vs_exact_recovery_summary(scored)
    overall = summary[summary["true_model"] == "overall"].iloc[0]

    assert events["event_id"].tolist() == ["evt-a", "evt-b"]
    assert [bool(value) for value in events["certified_vs_exact_recovered_expected_model"]] == [True, False]
    assert events["certified_vs_exact_reason"].tolist() == [
        "expected_lower_bound_beats_best_comparable",
        "expected_lower_bound_not_above_best_comparable",
    ]
    assert overall["simulated_events"] == 2
    assert overall["certified_vs_exact_recovered_events"] == 1
