from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.result_quality_gates import add_evidence_margin_columns, event_quality_summary


def test_result_quality_margins_are_position_scoped_with_duplicate_index() -> None:
    scores = pd.DataFrame(
        {
            "session": ["s0", "s0", "s0", "s0"],
            "event_index": [0, 0, 1, 1],
            "model": ["event0-high", "event0-low", "event1-low", "event1-high"],
            "log_evidence": [4.0, 1.0, 0.0, 5.0],
            "status": ["success", "success", "success", "success"],
            "diagnostic_candidate_evidence_support": [
                "exact_full_grid",
                "exact_full_grid",
                "exact_full_grid",
                "exact_full_grid",
            ],
        },
        index=[0, 1, 0, 1],
    )

    with_margins = add_evidence_margin_columns(scores)

    assert with_margins.index.tolist() == [0, 1, 0, 1]

    event0 = with_margins[with_margins["event_index"].eq(0)]
    assert event0["exact_model_best_model"].tolist() == ["event0-high", "event0-high"]
    assert event0["exact_model_rank"].tolist() == [1.0, 2.0]
    assert event0["exact_model_relative_log_evidence"].tolist() == pytest.approx([0.0, -3.0])

    event1 = with_margins[with_margins["event_index"].eq(1)]
    assert event1["exact_model_best_model"].tolist() == ["event1-high", "event1-high"]
    assert event1["exact_model_rank"].tolist() == [2.0, 1.0]
    assert event1["exact_model_relative_log_evidence"].tolist() == pytest.approx([-5.0, 0.0])

    summary = event_quality_summary(scores)
    best_by_event = dict(zip(summary["event_index"], summary["exact_best_model"], strict=True))
    assert best_by_event == {0: "event0-high", 1: "event1-high"}
