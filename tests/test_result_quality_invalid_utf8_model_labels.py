from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.result_quality_gates import (
    add_evidence_margin_columns,
    event_quality_summary,
    model_quality_summary,
)


def test_result_quality_preserves_distinct_invalid_utf8_model_labels() -> None:
    scores = pd.DataFrame(
        {
            "session": ["s0"] * 3,
            "event_index": [0] * 3,
            "model": pd.Series([b"\xff", b"\xfe", "\ufffd"], dtype=object),
            "log_evidence": [10.0, 4.0, 1.0],
            "status": ["success"] * 3,
            "diagnostic_candidate_evidence_support": ["exact_full_grid"] * 3,
        }
    )

    with_margins = add_evidence_margin_columns(scores)

    assert with_margins["exact_model_best_model"].tolist() == [
        "<invalid-utf8-bytes:ff>"
    ] * 3
    assert with_margins["exact_model_rank"].tolist() == [1.0, 2.0, 3.0]
    assert with_margins["exact_model_relative_log_evidence"].tolist() == pytest.approx(
        [0.0, -6.0, -9.0]
    )

    event_summary = event_quality_summary(scores)
    assert event_summary.loc[0, "exact_comparable_models"] == 3
    assert event_summary.loc[0, "exact_log_evidence_margin"] == pytest.approx(6.0)

    model_summary = model_quality_summary(scores)
    assert set(model_summary["model"]) == {
        "<invalid-utf8-bytes:ff>",
        "<invalid-utf8-bytes:fe>",
        "\ufffd",
    }
