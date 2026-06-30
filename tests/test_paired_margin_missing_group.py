from __future__ import annotations

import pandas as pd

from hipporeplayimm import advanced_result_diagnostics as diagnostics

GROUP_COLS = ("session", "event_index", "event_window_variant")
POSITIVE_MODEL = "sorted-spike-state-space-first-order-imm"
REFERENCE_MODEL = "sorted-spike-state-space-fragmented"


def test_paired_margin_decisions_keep_missing_optional_group_metadata() -> None:
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "event_window_variant": pd.NA,
                "status": "success",
                "evidence_comparable": True,
                "model": POSITIVE_MODEL,
                "log_evidence": 12.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "event_window_variant": pd.NA,
                "status": "success",
                "evidence_comparable": True,
                "model": REFERENCE_MODEL,
                "log_evidence": 10.0,
            },
        ]
    )

    decisions = diagnostics.paired_model_margin_decisions(
        scores,
        positive_model=POSITIVE_MODEL,
        reference_model=REFERENCE_MODEL,
        group_cols=GROUP_COLS,
    )

    assert len(decisions) == 1
    assert decisions["event_window_variant"].isna().all()
    assert decisions.loc[0, "margin_decision"] == POSITIVE_MODEL
    assert decisions.loc[0, "positive_minus_reference_log_evidence"] == 2.0
