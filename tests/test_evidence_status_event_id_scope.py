from __future__ import annotations

import pandas as pd

from hipporeplayimm.simulation_recovery import add_evidence_columns, expected_scoring_model


def _row(event_id: str, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "simulation_random_seed": 1,
        "simulation_event_index": 0,
        "event_index": 0,
        "event_id": event_id,
        "true_model": "diffusion",
        "expected_model": expected_scoring_model("diffusion"),
        "model": model,
        "log_evidence": log_evidence,
        "diagnostic_candidate_evidence_support": "truncated_full_grid",
    }


def test_lower_bound_recovery_flag_is_scoped_by_event_id() -> None:
    expected = expected_scoring_model("diffusion")
    alternative = "sorted-spike-state-space-stationary"
    rows = pd.DataFrame(
        [
            _row("evt-a", expected, 2.0),
            _row("evt-a", alternative, 0.0),
            _row("evt-b", expected, 0.0),
            _row("evt-b", alternative, 2.0),
        ]
    )

    scored = add_evidence_columns(rows)
    per_event = scored.groupby("event_id", sort=True).first()

    assert per_event["best_truncated_lower_bound_model"].to_dict() == {
        "evt-a": expected,
        "evt-b": alternative,
    }
    assert per_event["lower_bound_recovered_expected_model"].map(bool).to_dict() == {
        "evt-a": True,
        "evt-b": False,
    }
