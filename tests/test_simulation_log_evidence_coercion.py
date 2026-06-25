import pandas as pd

from hipporeplayimm.evidence_reporting import (
    TRUNCATED_EVIDENCE_SUPPORT,
    simulation_add_evidence_columns,
    simulation_event_best_rows,
)


def test_simulation_reporting_coerces_log_evidence_strings_before_ranking():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": "-2.0",
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-momentum-exact-sparse",
                "log_evidence": "not-a-number",
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "model": "sorted-spike-state-space-imm",
                "log_evidence": "1.0",
                "diagnostic_state_space_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
        ]
    )

    scored = simulation_add_evidence_columns(rows)
    diffusion = scored[scored["model"] == "sorted-spike-state-space-diffusion"].iloc[0]
    malformed_exact = scored[
        scored["model"] == "sorted-spike-state-space-momentum-exact-sparse"
    ].iloc[0]
    finite_lower_bound = scored[scored["model"] == "sorted-spike-state-space-imm"].iloc[0]

    assert bool(diffusion["is_best_model"])
    assert diffusion["model_probability"] == 1.0
    assert not bool(malformed_exact["evidence_comparable"])
    assert pd.isna(malformed_exact["log_evidence"])
    assert not bool(malformed_exact["is_best_model"])
    assert bool(finite_lower_bound["is_best_truncated_lower_bound"])


def test_simulation_event_best_rows_uses_numeric_log_evidence_for_csv_strings():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "low",
                "log_evidence": "2.0",
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "high",
                "log_evidence": "10.0",
            },
        ]
    )

    best = simulation_event_best_rows(rows)

    assert best["model"].tolist() == ["high"]
