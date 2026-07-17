from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_quality_gates import (
    MARGIN_DECISIVE,
    MARGIN_TIE,
    add_evidence_margin_columns,
    evidence_margin_label,
)


def test_evidence_margin_label_handles_arbitrary_precision_integers() -> None:
    huge = 10**400

    assert evidence_margin_label(huge) == MARGIN_DECISIVE
    assert evidence_margin_label(-huge) == MARGIN_TIE


def test_margin_annotation_ignores_unrepresentable_numeric_cells() -> None:
    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "best",
                "evidence_comparable": True,
                "evidence_support": "exact_full_grid",
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "second",
                "evidence_comparable": True,
                "evidence_support": "exact_full_grid",
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "overflow",
                "evidence_comparable": True,
                "evidence_support": "exact_full_grid",
            },
        ]
    )
    scores["log_evidence"] = pd.Series([8.0, 2.0, 10**400], dtype=object)

    annotated = add_evidence_margin_columns(scores)

    assert annotated["exact_model_best_model"].tolist() == ["best"] * 3
    assert annotated["exact_model_log_evidence_margin"].tolist() == [6.0] * 3
    np.testing.assert_array_equal(
        annotated["exact_model_rank"].to_numpy(),
        np.asarray([1.0, 2.0, np.nan]),
    )
