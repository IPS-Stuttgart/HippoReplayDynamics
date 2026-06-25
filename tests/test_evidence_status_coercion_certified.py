from __future__ import annotations

import pandas as pd

from hipporeplayimm.recovery_diagnostics import build_recovery_diagnostic_tables
from hipporeplayimm.simulation_recovery import (
    add_evidence_columns,
    certified_vs_exact_event_recovery,
    certified_vs_exact_recovery_summary,
    expected_scoring_model,
)


def test_certified_vs_exact_treats_missing_status_as_success() -> None:
    rows = pd.DataFrame(
        [
            _row(0, "momentum", "sorted-spike-state-space-diffusion", -2.0),
            _row(0, "momentum", "sorted-spike-state-space-momentum", -1.0),
        ]
    )
    scored = add_evidence_columns(rows)
    scored["status"] = pd.NA

    events = certified_vs_exact_event_recovery(scored)
    summary = certified_vs_exact_recovery_summary(scored)

    event = events.iloc[0]
    overall = summary[summary["true_model"].eq("overall")].iloc[0]
    assert bool(event["certified_vs_exact_recovered_expected_model"])
    assert event["certified_vs_exact_reason"] == "expected_comparable_best"
    assert overall["certified_vs_exact_recovered_events"] == 1
    assert overall["certified_vs_exact_recovery_accuracy"] == 1.0


def test_recovery_diagnostics_use_status_coerced_certified_recovery() -> None:
    rows = pd.DataFrame(
        [
            _row(0, "momentum", "sorted-spike-state-space-diffusion", -2.0),
            _row(0, "momentum", "sorted-spike-state-space-momentum", -1.0),
        ]
    )
    scored = add_evidence_columns(rows)
    scored["status"] = ""

    tables = build_recovery_diagnostic_tables(scored)
    event = tables.event_diagnostics.iloc[0]
    overall = tables.summary[tables.summary["true_model"].eq("overall")].iloc[0]

    assert event["successful_scores"] == 2
    assert event["comparable_scores"] == 2
    assert bool(event["strict_recovered_expected_model"])
    assert bool(event["certified_vs_exact_recovered_expected_model"])
    assert event["certified_vs_exact_reason"] == "expected_comparable_best"
    assert overall["strict_recovered_events"] == 1
    assert overall["certified_vs_exact_recovered_events"] == 1


def _row(event_index: int, true_model: str, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "true_model": true_model,
        "expected_model": expected_scoring_model(true_model),
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
    }
