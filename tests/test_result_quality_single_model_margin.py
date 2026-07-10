from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import EXACT_EVIDENCE_SUPPORT
from hipporeplayimm.result_quality_gates import (
    MARGIN_DECISIVE,
    MARGIN_UNKNOWN,
    add_evidence_margin_columns,
    event_quality_summary,
    quality_gate_summary,
)


def _single_and_compared_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "only-model",
                "log_evidence": 5.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": True,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "winner",
                "log_evidence": 12.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": True,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "runner-up",
                "log_evidence": 0.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": True,
            },
        ]
    )


def test_single_finite_model_has_unknown_not_decisive_margin() -> None:
    annotated = add_evidence_margin_columns(_single_and_compared_events())
    only = annotated[annotated["event_index"].eq(0)]

    assert only["exact_model_best_model"].tolist() == ["only-model"]
    assert only["exact_model_rank"].tolist() == [1.0]
    assert only["exact_model_relative_log_evidence"].tolist() == [0.0]
    assert only["exact_model_margin_category"].tolist() == [MARGIN_UNKNOWN]
    assert only["exact_model_log_evidence_margin"].isna().all()


def test_single_model_event_does_not_inflate_strong_margin_fraction() -> None:
    scores = _single_and_compared_events()
    summary = event_quality_summary(scores).set_index("event_index")

    assert summary.loc[0, "exact_margin_category"] == MARGIN_UNKNOWN
    assert np.isnan(summary.loc[0, "exact_log_evidence_margin"])
    assert summary.loc[1, "exact_margin_category"] == MARGIN_DECISIVE

    gates = quality_gate_summary(scores).set_index("gate")
    assert gates.loc["strong_exact_margin_fraction", "value"] == 0.5
