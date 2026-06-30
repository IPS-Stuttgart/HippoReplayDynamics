import pandas as pd

from hipporeplayimm.recovery_diagnostics import build_recovery_diagnostic_tables
from hipporeplayimm.simulation_recovery import expected_scoring_model


def _row(source: str, seed: int, log_evidence: float) -> dict[str, object]:
    true_model = "stationary"
    model = expected_scoring_model(true_model)
    return {
        "status": "success",
        "session": "Rat1/Open1",
        "source_recovery_score_file": source,
        "simulation_random_seed": seed,
        "simulation_event_index": 0,
        "event_index": 0,
        "true_model": true_model,
        "expected_model": model,
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 4,
        "n_spikes": 7,
    }


def test_recovery_diagnostics_keep_independent_run_event_scope():
    scores = pd.DataFrame(
        [
            _row("run-a/simulation_recovery_event_scores.csv", 11, -1.0),
            _row("run-b/simulation_recovery_event_scores.csv", 22, -2.0),
        ]
    )

    tables = build_recovery_diagnostic_tables(scores)

    assert len(tables.event_diagnostics) == 2
    assert tables.manifest["n_diagnostic_events"] == 2
    assert tables.event_diagnostics["source_recovery_score_file"].tolist() == [
        "run-a/simulation_recovery_event_scores.csv",
        "run-b/simulation_recovery_event_scores.csv",
    ]
    assert tables.event_diagnostics["simulation_random_seed"].tolist() == [11, 22]

    overall = tables.certified_vs_exact_summary[
        tables.certified_vs_exact_summary["true_model"] == "overall"
    ].iloc[0]
    assert overall["simulated_events"] == 2
