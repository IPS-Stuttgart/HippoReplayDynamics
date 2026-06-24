from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.result_quality_gates import (
    add_evidence_margin_columns,
    event_quality_summary,
    quality_gate_summary,
)


def test_result_quality_summaries_treat_group_less_scores_as_one_event() -> None:
    scores = pd.DataFrame(
        {
            "model": ["diffusion", "stationary"],
            "log_evidence": [4.0, 1.5],
            "status": ["success", "success"],
            "diagnostic_candidate_evidence_support": ["exact_full_grid", "exact_full_grid"],
        }
    )

    with_margins = add_evidence_margin_columns(scores)

    assert with_margins["exact_model_best_model"].tolist() == ["diffusion", "diffusion"]
    assert with_margins["exact_model_rank"].tolist() == [1.0, 2.0]
    assert with_margins["exact_model_log_evidence_margin"].tolist() == pytest.approx([2.5, 2.5])

    event_summary = event_quality_summary(scores)
    assert event_summary.shape[0] == 1
    assert int(event_summary.loc[0, "score_rows"]) == 2
    assert event_summary.loc[0, "exact_best_model"] == "diffusion"
    assert float(event_summary.loc[0, "exact_log_evidence_margin"]) == pytest.approx(2.5)

    gate_summary = quality_gate_summary(scores)
    exact_gate = gate_summary.loc[gate_summary["gate"].eq("events_with_min_exact_models")].iloc[0]
    assert exact_gate["status"] == "pass"
    assert float(exact_gate["value"]) == pytest.approx(1.0)
