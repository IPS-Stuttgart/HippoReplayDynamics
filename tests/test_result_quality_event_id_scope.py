from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_quality_gates import (
    add_evidence_margin_columns,
    event_group_columns,
    event_quality_summary,
)


def test_result_quality_uses_event_id_when_event_index_is_missing() -> None:
    scores = pd.DataFrame(
        {
            "session": ["RatX/OpenY", "RatX/OpenY", "RatX/OpenY", "RatX/OpenY"],
            "event_id": [0, 0, 1, 1],
            "model": ["diffusion", "stationary", "diffusion", "stationary"],
            "log_evidence": [4.0, 1.0, 0.5, 3.0],
            "status": ["success", "success", "success", "success"],
            "diagnostic_candidate_evidence_support": [
                "exact_full_grid",
                "exact_full_grid",
                "exact_full_grid",
                "exact_full_grid",
            ],
        }
    )

    assert event_group_columns(scores) == ["session", "event_id"]

    with_margins = add_evidence_margin_columns(scores)
    assert with_margins["exact_model_best_model"].tolist() == [
        "diffusion",
        "diffusion",
        "stationary",
        "stationary",
    ]

    event_summary = event_quality_summary(scores).sort_values("event_id").reset_index(drop=True)
    assert event_summary.shape[0] == 2
    assert event_summary["score_rows"].tolist() == [2, 2]
    assert event_summary["exact_best_model"].tolist() == ["diffusion", "stationary"]


def test_result_quality_prefers_event_index_over_event_id() -> None:
    scores = pd.DataFrame(
        {
            "session": ["RatX/OpenY", "RatX/OpenY"],
            "event_index": [4, 4],
            "event_id": [99, 100],
            "model": ["diffusion", "stationary"],
            "log_evidence": [2.0, 1.0],
            "status": ["success", "success"],
            "diagnostic_candidate_evidence_support": ["exact_full_grid", "exact_full_grid"],
        }
    )

    assert event_group_columns(scores) == ["session", "event_index"]
