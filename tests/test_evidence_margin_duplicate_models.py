from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import add_evidence_margin_columns, evidence_margin_table


def test_evidence_margin_table_ignores_duplicate_best_model_as_runner_up() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0],
            "model": ["momentum", "momentum", "diffusion"],
            "log_evidence": [12.0, 11.0, 2.0],
            "status": ["success", "success", "success"],
            "evidence_comparable": [True, True, True],
        }
    )

    margins = evidence_margin_table(scores)

    assert margins.loc[0, "best_model_by_evidence"] == "momentum"
    assert margins.loc[0, "second_best_model_by_evidence"] == "diffusion"
    assert margins.loc[0, "best_log_evidence"] == 12.0
    assert margins.loc[0, "second_best_log_evidence"] == 2.0
    assert margins.loc[0, "evidence_margin_to_second_best"] == 10.0
    assert margins.loc[0, "models_compared"] == 2


def test_add_evidence_margin_columns_uses_distinct_model_margin() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0],
            "model": ["stationary", "stationary", "diffusion"],
            "log_evidence": [5.0, 4.5, 1.0],
            "status": ["success", "success", "success"],
            "evidence_comparable": [True, True, True],
        }
    )

    merged = add_evidence_margin_columns(scores)

    assert merged["second_best_model_by_evidence"].unique().tolist() == ["diffusion"]
    assert merged["evidence_margin_to_second_best"].unique().tolist() == [4.0]


def test_add_evidence_margin_columns_preserves_named_duplicate_index() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1", None, None],
            "event_index": [0, 0, 0, 1, 1],
            "model": ["momentum", "momentum", "diffusion", "stationary", "diffusion"],
            "log_evidence": [12.0, 11.0, 2.0, 5.0, 1.0],
            "status": ["success"] * 5,
            "evidence_comparable": [True] * 5,
        },
        index=pd.Index(
            ["winner", "winner", "runner-up", "missing-group", "missing-group"],
            name="score_index",
        ),
    )

    merged = add_evidence_margin_columns(scores)

    pd.testing.assert_index_equal(merged.index, scores.index)
    assert merged["evidence_margin_to_second_best"].tolist() == [10.0, 10.0, 10.0, 4.0, 4.0]
