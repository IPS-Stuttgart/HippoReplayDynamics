import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
    simulation_add_evidence_columns,
)


def test_state_space_imm_support_diagnostic_is_truncated_not_exact():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "expected_model": "sorted-spike-state-space-diffusion",
                "log_evidence": -1.0,
                "n_time": 3,
                "n_spikes": 5,
            },
            {
                "status": "success",
                "session": "RatX/OpenY",
                "event_index": 0,
                "model": "sorted-spike-state-space-imm",
                "expected_model": "sorted-spike-state-space-diffusion",
                "log_evidence": 100.0,
                "n_time": 3,
                "n_spikes": 5,
                "diagnostic_state_space_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
        ]
    )

    scored = simulation_add_evidence_columns(rows)
    diffusion = scored[scored["model"] == "sorted-spike-state-space-diffusion"].iloc[0]
    imm = scored[scored["model"] == "sorted-spike-state-space-imm"].iloc[0]

    assert diffusion["evidence_support"] == EXACT_EVIDENCE_SUPPORT
    assert bool(diffusion["evidence_comparable"])
    assert bool(diffusion["is_best_model"])
    assert diffusion["best_model"] == "sorted-spike-state-space-diffusion"

    assert imm["evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert not bool(imm["evidence_comparable"])
    assert not bool(imm["is_best_model"])
    assert pd.isna(imm["model_probability"])
    assert imm["truncated_relative_log_evidence"] == 0.0
    assert bool(imm["is_best_truncated_lower_bound"])
    assert imm["best_truncated_lower_bound_model"] == "sorted-spike-state-space-imm"


def test_state_space_imm_support_column_is_used_by_generic_inference():
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "sorted-spike-state-space-imm",
                "log_evidence": 10.0,
                "diagnostic_state_space_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            }
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert not bool(scored.loc[0, "evidence_comparable"])
