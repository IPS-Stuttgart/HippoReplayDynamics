from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT
from hipporeplayimm.result_quality_gates import (
    MARGIN_DECISIVE,
    add_evidence_margin_columns,
)


def test_truncated_margin_ranking_ignores_nonfinite_log_evidence() -> None:
    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "bad-infinite-lower-bound",
                "log_evidence": np.inf,
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
                "evidence_comparable": False,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "finite-best-lower-bound",
                "log_evidence": 3.0,
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
                "evidence_comparable": False,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "finite-second-lower-bound",
                "log_evidence": 1.0,
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
                "evidence_comparable": False,
            },
        ]
    )

    annotated = add_evidence_margin_columns(scores)
    by_model = annotated.set_index("model")

    assert set(annotated["truncated_lower_bound_best_model"]) == {"finite-best-lower-bound"}
    assert by_model.loc["finite-best-lower-bound", "truncated_lower_bound_rank"] == 1.0
    assert by_model.loc["finite-second-lower-bound", "truncated_lower_bound_rank"] == 2.0
    assert pd.isna(by_model.loc["bad-infinite-lower-bound", "truncated_lower_bound_rank"])
    assert set(annotated["truncated_lower_bound_log_evidence_margin"]) == {2.0}


def test_exact_margin_keeps_negative_infinite_comparator() -> None:
    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "finite",
                "log_evidence": 3.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": True,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "impossible",
                "log_evidence": -np.inf,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": True,
            },
        ]
    )

    annotated = add_evidence_margin_columns(scores)
    by_model = annotated.set_index("model")

    assert annotated["exact_model_best_model"].eq("finite").all()
    assert by_model.loc["finite", "exact_model_rank"] == 1.0
    assert by_model.loc["impossible", "exact_model_rank"] == 2.0
    assert np.isneginf(
        by_model.loc["impossible", "exact_model_relative_log_evidence"]
    )
    assert np.isposinf(annotated["exact_model_log_evidence_margin"]).all()
    assert annotated["exact_model_margin_category"].eq(MARGIN_DECISIVE).all()
