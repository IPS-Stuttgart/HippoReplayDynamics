from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.advanced_result_diagnostics import (
    add_evidence_margin_columns,
    evidence_margin_table,
)


def _matrix_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0] * 4,
            "matrix_id": ["slow", "slow", "fast", "fast"],
            "model": ["diffusion", "stationary", "diffusion", "stationary"],
            "log_evidence": [10.0, 0.0, 1.0, 9.0],
            "status": ["success"] * 4,
            "evidence_comparable": [True] * 4,
        },
        index=[101, 102, 201, 202],
    )


def test_evidence_margin_table_keeps_matrix_cells_separate() -> None:
    margins = evidence_margin_table(_matrix_scores()).set_index("matrix_id")

    assert margins.index.tolist() == ["slow", "fast"]
    assert margins.loc["slow", "best_model_by_evidence"] == "diffusion"
    assert margins.loc["fast", "best_model_by_evidence"] == "stationary"
    assert margins.loc["slow", "evidence_margin_to_second_best"] == pytest.approx(10.0)
    assert margins.loc["fast", "evidence_margin_to_second_best"] == pytest.approx(8.0)
    assert margins["models_compared"].tolist() == [2, 2]


def test_add_evidence_margin_columns_preserves_matrix_scope_and_index() -> None:
    scores = _matrix_scores()

    annotated = add_evidence_margin_columns(scores)

    assert annotated.index.equals(scores.index)
    assert len(annotated) == len(scores)
    assert annotated.loc[[101, 102], "best_model_by_evidence"].tolist() == [
        "diffusion",
        "diffusion",
    ]
    assert annotated.loc[[201, 202], "best_model_by_evidence"].tolist() == [
        "stationary",
        "stationary",
    ]
    assert annotated.loc[[101, 102], "evidence_margin_to_second_best"].tolist() == pytest.approx(
        [10.0, 10.0]
    )
    assert annotated.loc[[201, 202], "evidence_margin_to_second_best"].tolist() == pytest.approx(
        [8.0, 8.0]
    )
