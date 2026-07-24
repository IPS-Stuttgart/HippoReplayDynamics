from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_quality_gates import (
    add_evidence_margin_columns,
    event_quality_summary,
    model_quality_summary,
    quality_gate_summary,
)


def _byte_backed_model_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["s0"] * 6,
            "event_index": [0] * 6,
            "model": pd.Series(
                [
                    b"winner",
                    bytearray(b"winner"),
                    memoryview(b"winner"),
                    np.bytes_(b"winner"),
                    "winner",
                    "runner-up",
                ],
                dtype=object,
            ),
            "log_evidence": [7.0, 8.0, 9.0, 6.0, 10.0, 4.0],
            "status": ["success"] * 6,
            "diagnostic_candidate_evidence_support": ["exact_full_grid"] * 6,
        }
    )


def test_result_quality_margins_merge_byte_backed_model_labels() -> None:
    scores = _byte_backed_model_scores()

    with_margins = add_evidence_margin_columns(scores)

    assert with_margins["exact_model_best_model"].tolist() == ["winner"] * 6
    assert with_margins["exact_model_log_evidence_margin"].tolist() == pytest.approx(
        [6.0] * 6
    )
    assert with_margins["exact_model_margin_category"].tolist() == ["strong"] * 6
    assert with_margins["exact_model_rank"].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    assert with_margins["exact_model_relative_log_evidence"].tolist() == pytest.approx(
        [-3.0, -2.0, -1.0, -4.0, 0.0, -6.0]
    )

    event_summary = event_quality_summary(scores)
    assert event_summary.loc[0, "exact_comparable_models"] == 2
    assert event_summary.loc[0, "exact_best_model"] == "winner"
    assert event_summary.loc[0, "exact_log_evidence_margin"] == pytest.approx(6.0)


def test_result_quality_summaries_group_unhashable_byte_labels() -> None:
    scores = _byte_backed_model_scores()

    model_summary = model_quality_summary(scores).set_index("model")

    assert set(model_summary.index) == {"winner", "runner-up"}
    assert model_summary.loc["winner", "rows"] == 5
    assert model_summary.loc["runner-up", "rows"] == 1

    gates = quality_gate_summary(scores, min_exact_models_per_event=3).set_index("gate")
    assert gates.loc["events_with_min_exact_models", "status"] == "warn"
    assert gates.loc["events_with_min_exact_models", "value"] == pytest.approx(0.0)
