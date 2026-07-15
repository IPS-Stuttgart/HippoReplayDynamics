from __future__ import annotations

import pandas as pd

import hipporeplayimm.advanced_result_diagnostics as diagnostics
from hipporeplayimm.advanced_result_evidence_margin_duplicates import (
    apply_evidence_margin_distinct_model_patch,
)


def test_evidence_margin_table_handles_arbitrary_size_integer_flags() -> None:
    huge = 10**400
    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "best",
                "log_evidence": 8.0,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "second",
                "log_evidence": 2.0,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "excluded",
                "log_evidence": 100.0,
            },
        ]
    )
    scores["evidence_comparable"] = pd.Series([huge, -huge, 0], dtype=object)

    margins = diagnostics.evidence_margin_table(scores)

    assert margins["best_model_by_evidence"].tolist() == ["best"]
    assert margins["second_best_model_by_evidence"].tolist() == ["second"]
    assert margins["evidence_margin_to_second_best"].tolist() == [6.0]
    assert margins["evidence_margin_category"].tolist() == ["strong"]
    assert margins["models_compared"].tolist() == [2]


def test_evidence_margin_integer_flag_patch_is_idempotent() -> None:
    patched = diagnostics._as_bool

    apply_evidence_margin_distinct_model_patch()

    assert diagnostics._as_bool is patched
