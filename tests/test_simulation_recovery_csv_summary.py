from __future__ import annotations

import pandas as pd

from hipporeplayimm.simulation_recovery import add_evidence_columns, expected_scoring_model, recovery_summary


def test_recovery_summary_coerces_csv_bool_strings_and_counts_session_events() -> None:
    rows = pd.DataFrame(
        [
            _row("Rat1/Open1", 0, "stationary", "sorted-spike-state-space-stationary", -1.0),
            _row("Rat2/Open1", 0, "diffusion", "sorted-spike-state-space-stationary", -1.0),
        ]
    )
    scored = add_evidence_columns(rows)
    scored["recovered_expected_model"] = scored["recovered_expected_model"].map(
        lambda value: "True" if bool(value) else "False"
    )

    summary = recovery_summary(scored)
    overall = summary[summary["true_model"] == "overall"].iloc[0]

    assert overall["simulated_events"] == 2
    assert overall["recovered_events"] == 1
    assert overall["recovery_accuracy"] == 0.5


def _row(session: str, event_index: int, true_model: str, model: str, log_evidence: float) -> dict[str, object]:
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
