from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    paired_model_margin_decisions,
    paired_model_margin_summary,
)


def test_zero_threshold_equal_evidence_is_ambiguous_not_positive_claim():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "true_model": ["diffusion", "diffusion"],
            "model": [
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum-exact-sparse",
            ],
            "log_evidence": [5.0, 5.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )

    decisions = paired_model_margin_decisions(
        scores,
        positive_model="sorted-spike-state-space-momentum-exact-sparse",
        reference_model="sorted-spike-state-space-diffusion",
        margin_threshold=0.0,
        true_model_col="true_model",
        positive_true_label="momentum",
    )
    summary = paired_model_margin_summary(decisions, true_model_col="true_model")

    assert decisions.loc[0, "positive_minus_reference_log_evidence"] == 0.0
    assert decisions.loc[0, "margin_decision"] == "ambiguous"
    assert not bool(decisions.loc[0, "positive_model_claimed"])
    assert bool(decisions.loc[0, "margin_binary_correct"])
    assert summary.loc[0, "positive_model_claims"] == 0
    assert summary.loc[0, "false_positive_claims"] == 0
    assert summary.loc[0, "ambiguous_events"] == 1
