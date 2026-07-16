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
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0, 0, 1, 1],
            "model": ["stationary", "diffusion", "momentum", "stationary"],
            "log_evidence": [5.0, 1.0, 8.0, 7.0],
            "status": ["success"] * 4,
            "evidence_comparable": [True] * 4,
        },
        index=pd.Index(
            ["score-row", "score-row", "other-row", "other-row"],
            name="score_index",
        ),
    )

    annotated = add_evidence_margin_columns(scores)

    pd.testing.assert_index_equal(annotated.index, scores.index)
    assert annotated["best_model_by_evidence"].tolist() == [
        "stationary",
        "stationary",
        "momentum",
        "momentum",
    ]
    assert annotated["evidence_margin_to_second_best"].tolist() == [4.0, 4.0, 1.0, 1.0]


def test_add_evidence_margin_columns_refreshes_existing_annotations() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "diffusion"],
            "log_evidence": [5.0, 1.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )
    annotated = add_evidence_margin_columns(scores)
    annotated.loc[annotated["model"] == "diffusion", "log_evidence"] = 4.0

    refreshed = add_evidence_margin_columns(annotated)

    assert refreshed["evidence_margin_to_second_best"].tolist() == [1.0, 1.0]
    assert refreshed["second_best_model_by_evidence"].tolist() == [
        "diffusion",
        "diffusion",
    ]
    assert not any(column.endswith(("_x", "_y")) for column in refreshed.columns)
    for column in (
        "best_model_by_evidence",
        "second_best_model_by_evidence",
        "best_log_evidence",
        "second_best_log_evidence",
        "evidence_margin_to_second_best",
        "evidence_margin_category",
        "models_compared",
    ):
        assert refreshed.columns.tolist().count(column) == 1
