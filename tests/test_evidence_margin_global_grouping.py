from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    add_evidence_margin_columns,
    evidence_margin_table,
)


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["momentum", "momentum", "diffusion"],
            "log_evidence": [12.0, 11.0, 2.0],
            "status": ["success", "success", "success"],
            "evidence_comparable": [True, True, True],
        },
        index=pd.Index(["first", "duplicate", "runner-up"], name="score_row"),
    )


def test_evidence_margin_table_supports_table_wide_grouping() -> None:
    margins = evidence_margin_table(_scores(), group_cols=())

    assert margins.to_dict("records") == [
        {
            "best_model_by_evidence": "momentum",
            "second_best_model_by_evidence": "diffusion",
            "best_log_evidence": 12.0,
            "second_best_log_evidence": 2.0,
            "evidence_margin_to_second_best": 10.0,
            "evidence_margin_category": "strong",
            "models_compared": 2,
        }
    ]


def test_add_evidence_margin_columns_broadcasts_table_wide_margin() -> None:
    scores = _scores()

    annotated = add_evidence_margin_columns(scores, group_cols=())

    pd.testing.assert_index_equal(annotated.index, scores.index)
    assert annotated["best_model_by_evidence"].tolist() == ["momentum"] * 3
    assert annotated["second_best_model_by_evidence"].tolist() == ["diffusion"] * 3
    assert annotated["evidence_margin_to_second_best"].tolist() == [10.0] * 3
    assert annotated["models_compared"].tolist() == [2] * 3
