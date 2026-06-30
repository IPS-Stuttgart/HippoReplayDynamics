from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import paired_model_margin_decisions
from hipporeplayimm.posterior_calibration_summary_patch import _MISSING_GROUP_SENTINEL


def test_paired_model_missing_group_patch_preserves_literal_sentinel_group_value() -> None:
    rows = []
    for event_index, positive_log_evidence, reference_log_evidence in (
        (_MISSING_GROUP_SENTINEL, 10.0, 0.0),
        (pd.NA, 2.0, 1.0),
    ):
        rows.extend(
            [
                {
                    "session": "Rat1/Open1",
                    "event_index": event_index,
                    "model": "positive-model",
                    "log_evidence": positive_log_evidence,
                    "status": "success",
                    "evidence_comparable": True,
                },
                {
                    "session": "Rat1/Open1",
                    "event_index": event_index,
                    "model": "reference-model",
                    "log_evidence": reference_log_evidence,
                    "status": "success",
                    "evidence_comparable": True,
                },
            ]
        )
    scores = pd.DataFrame(rows)

    decisions = paired_model_margin_decisions(
        scores,
        positive_model="positive-model",
        reference_model="reference-model",
    )

    assert len(decisions) == 2
    literal_rows = decisions[decisions["event_index"].astype(object).eq(_MISSING_GROUP_SENTINEL)]
    assert len(literal_rows) == 1
    missing_rows = decisions[decisions["event_index"].isna()]
    assert len(missing_rows) == 1
    assert literal_rows.iloc[0]["positive_minus_reference_log_evidence"] == 10.0
    assert missing_rows.iloc[0]["positive_minus_reference_log_evidence"] == 1.0
