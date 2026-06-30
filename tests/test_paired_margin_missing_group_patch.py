from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    paired_model_margin_decisions,
    paired_model_margin_threshold_sweep,
)


def test_paired_model_margin_decisions_keep_missing_optional_group_metadata() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [4, 4],
            "event_window_variant": [pd.NA, pd.NA],
            "true_model": ["momentum", "momentum"],
            "model": ["momentum", "diffusion"],
            "log_evidence": [5.0, 1.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )
    group_cols = ("session", "event_index", "event_window_variant")

    decisions = paired_model_margin_decisions(
        scores,
        positive_model="momentum",
        reference_model="diffusion",
        margin_threshold=2.0,
        group_cols=group_cols,
        true_model_col="true_model",
        positive_true_label="momentum",
    )
    sweep = paired_model_margin_threshold_sweep(
        scores,
        positive_model="momentum",
        reference_model="diffusion",
        thresholds=(2.0,),
        group_cols=group_cols,
        true_model_col="true_model",
        positive_true_label="momentum",
    )

    assert len(decisions) == 1
    assert decisions["event_window_variant"].isna().all()
    assert decisions.loc[0, "margin_decision"] == "momentum"
    assert bool(decisions.loc[0, "positive_model_claimed"]) is True
    assert bool(decisions.loc[0, "margin_binary_correct"]) is True
    assert int(sweep.loc[0, "events"]) == 1
    assert float(sweep.loc[0, "positive_claim_recall"]) == 1.0
