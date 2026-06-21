import pandas as pd

from hipporeplayimm.simulation_recovery import (
    add_evidence_columns,
    certified_vs_exact_recovery_summary,
    expected_scoring_model,
    recovery_summary,
)


def test_recovery_summaries_count_session_event_pairs_with_overlapping_indices():
    rows = pd.DataFrame(
        [
            _row("Rat1/Open1", 0, "stationary", "sorted-spike-state-space-stationary", -1.0),
            _row("Rat1/Open1", 0, "stationary", "sorted-spike-state-space-diffusion", -3.0),
            _row("Rat2/Open1", 0, "stationary", "sorted-spike-state-space-stationary", -2.0),
            _row("Rat2/Open1", 0, "stationary", "sorted-spike-state-space-diffusion", -4.0),
        ]
    )

    scored = add_evidence_columns(rows)
    summary = recovery_summary(scored)
    certified = certified_vs_exact_recovery_summary(scored)

    stationary = summary[summary["true_model"] == "stationary"].iloc[0]
    overall = summary[summary["true_model"] == "overall"].iloc[0]
    certified_stationary = certified[certified["true_model"] == "stationary"].iloc[0]
    certified_overall = certified[certified["true_model"] == "overall"].iloc[0]

    assert stationary["simulated_events"] == 2
    assert overall["simulated_events"] == 2
    assert stationary["recovered_events"] == 2
    assert certified_stationary["simulated_events"] == 2
    assert certified_overall["simulated_events"] == 2
    assert certified_stationary["certified_vs_exact_recovered_events"] == 2


def _row(
    session: str,
    event_index: int,
    true_model: str,
    model: str,
    log_evidence: float,
) -> dict[str, object]:
    return {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "true_model": true_model,
        "expected_model": expected_scoring_model(true_model),
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
    }
