from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    add_evidence_margin_columns,
    evidence_margin_table,
)


def _scores_with_invalid_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 6,
            "event_index": [0] * 6,
            "model": ["invalid-positive", "invalid-negative", "a", "a", "b", "invalid-text"],
            "log_evidence": [np.inf, -np.inf, "5.0", 4.0, 3.0, "not-a-number"],
            "status": ["success"] * 6,
            "evidence_comparable": [True] * 6,
        }
    )


def test_evidence_margin_ignores_nonfinite_and_nonnumeric_rows() -> None:
    margins = evidence_margin_table(_scores_with_invalid_evidence())

    assert len(margins) == 1
    margin = margins.iloc[0]
    assert margin["best_model_by_evidence"] == "a"
    assert margin["second_best_model_by_evidence"] == "b"
    assert margin["best_log_evidence"] == 5.0
    assert margin["second_best_log_evidence"] == 3.0
    assert margin["evidence_margin_to_second_best"] == 2.0
    assert margin["evidence_margin_category"] == "weak"
    assert margin["models_compared"] == 2


def test_evidence_margin_columns_use_only_finite_model_scores() -> None:
    scores = _scores_with_invalid_evidence()

    merged = add_evidence_margin_columns(scores)

    assert merged["best_model_by_evidence"].eq("a").all()
    assert merged["second_best_model_by_evidence"].eq("b").all()
    assert merged["evidence_margin_to_second_best"].eq(2.0).all()
