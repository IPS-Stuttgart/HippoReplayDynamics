from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import paired_model_margin_decisions
from scripts.advanced_result_diagnostics import _paired_group_cols_from_arg


def test_auto_paired_grouping_keeps_sweep_scopes_separate() -> None:
    scores = pd.DataFrame(
        [
            {
                "matrix_id": "cell-a",
                "random_seed": 11,
                "session": "Rat1/Open1",
                "event_index": 3,
                "model": "positive",
                "log_evidence": 12.0,
                "status": "success",
            },
            {
                "matrix_id": "cell-a",
                "random_seed": 11,
                "session": "Rat1/Open1",
                "event_index": 3,
                "model": "reference",
                "log_evidence": 2.0,
                "status": "success",
            },
            {
                "matrix_id": "cell-b",
                "random_seed": 22,
                "session": "Rat1/Open1",
                "event_index": 3,
                "model": "positive",
                "log_evidence": 1.0,
                "status": "success",
            },
            {
                "matrix_id": "cell-b",
                "random_seed": 22,
                "session": "Rat1/Open1",
                "event_index": 3,
                "model": "reference",
                "log_evidence": 9.0,
                "status": "success",
            },
        ]
    )

    group_cols = _paired_group_cols_from_arg("auto", scores)

    assert group_cols == ("matrix_id", "random_seed", "session", "event_index")
    decisions = paired_model_margin_decisions(
        scores,
        positive_model="positive",
        reference_model="reference",
        margin_threshold=1.0,
        group_cols=group_cols,
    )
    assert len(decisions) == 2
    assert set(decisions["matrix_id"]) == {"cell-a", "cell-b"}
    assert set(decisions["margin_decision"]) == {"positive", "reference"}


def test_explicit_paired_grouping_override_is_preserved() -> None:
    scores = pd.DataFrame(
        {
            "matrix_id": ["cell-a"],
            "random_seed": [11],
            "session": ["Rat1/Open1"],
            "event_index": [3],
        }
    )

    assert _paired_group_cols_from_arg("session,event_index", scores) == (
        "session",
        "event_index",
    )
