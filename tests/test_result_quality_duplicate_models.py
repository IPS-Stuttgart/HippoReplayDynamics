from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.result_quality_gates import (
    add_evidence_margin_columns,
    event_quality_summary,
)


def test_result_quality_margins_compare_distinct_models() -> None:
    scores = pd.DataFrame(
        {
            "session": ["s0", "s0", "s0"],
            "event_index": [0, 0, 0],
            "model": ["winner", "winner", "runner-up"],
            "log_evidence": [10.0, 9.0, 4.0],
            "status": ["success", "success", "success"],
            "diagnostic_candidate_evidence_support": [
                "exact_full_grid",
                "exact_full_grid",
                "exact_full_grid",
            ],
        }
    )

    with_margins = add_evidence_margin_columns(scores)

    assert with_margins["exact_model_best_model"].tolist() == [
        "winner",
        "winner",
        "winner",
    ]
    assert with_margins["exact_model_log_evidence_margin"].tolist() == pytest.approx(
        [6.0, 6.0, 6.0]
    )
    assert with_margins["exact_model_margin_category"].tolist() == [
        "strong",
        "strong",
        "strong",
    ]
    assert with_margins["exact_model_rank"].tolist() == [1.0, 1.0, 2.0]
    assert with_margins["exact_model_relative_log_evidence"].tolist() == pytest.approx(
        [0.0, -1.0, -6.0]
    )

    summary = event_quality_summary(scores)

    assert summary.loc[0, "exact_comparable_models"] == 2
    assert summary.loc[0, "exact_best_model"] == "winner"
    assert summary.loc[0, "exact_log_evidence_margin"] == pytest.approx(6.0)
    assert summary.loc[0, "exact_margin_category"] == "strong"
