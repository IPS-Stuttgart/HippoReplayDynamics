from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import evidence_margin_table


def test_evidence_margin_table_compares_distinct_models_when_rows_repeat() -> None:
    scores = pd.DataFrame(
        {
            "session": ["s", "s", "s"],
            "event_index": [0, 0, 0],
            "model": ["a", "a", "b"],
            "log_evidence": [10.0, 9.0, 4.0],
            "status": ["success", "success", "success"],
            "evidence_comparable": [True, True, True],
        }
    )

    margin = evidence_margin_table(scores).iloc[0]

    assert margin["best_model_by_evidence"] == "a"
    assert margin["second_best_model_by_evidence"] == "b"
    assert margin["evidence_margin_to_second_best"] == 6.0
    assert margin["models_compared"] == 2
