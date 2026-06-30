from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.simulation_recovery import certified_vs_exact_event_recovery, expected_scoring_model


def _row(true_model: str, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "simulation_random_seed": 1,
        "simulation_event_index": 0,
        "event_index": 0,
        "true_model": true_model,
        "expected_model": expected_scoring_model(true_model),
        "model": model,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 5,
        "diagnostic_candidate_evidence_support": "exact_full_grid",
    }


def test_certified_recovery_uses_winning_duplicate_model_row() -> None:
    expected = expected_scoring_model("stationary")
    rows = pd.DataFrame(
        [
            _row("stationary", expected, 0.0),
            _row("stationary", expected, 10.0),
            _row("stationary", "sorted-spike-state-space-diffusion", 5.0),
        ]
    )

    event = certified_vs_exact_event_recovery(rows).iloc[0]

    assert event["best_comparable_model"] == expected
    assert event["certified_reference_model"] == expected
    assert event["expected_model_log_evidence"] == pytest.approx(10.0)
    assert event["best_comparable_log_evidence"] == pytest.approx(10.0)
    assert event["expected_minus_best_comparable_log_evidence"] == pytest.approx(0.0)
    assert bool(event["expected_model_evidence_comparable"])
    assert bool(event["certified_vs_exact_recovered_expected_model"])
