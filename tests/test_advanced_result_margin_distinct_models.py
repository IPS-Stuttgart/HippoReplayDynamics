from __future__ import annotations

import pandas as pd
import pytest

import hipporeplayimm as _package  # noqa: F401
from hipporeplayimm import advanced_result_diagnostics as diagnostics


def test_runtime_patch_uses_distinct_second_best_model():
    scores = pd.DataFrame(
        [
            dict(session="s", event_index=0, model="a", log_evidence=5.0, status="success", evidence_comparable=True),
            dict(session="s", event_index=0, model="a", log_evidence=4.9, status="success", evidence_comparable=True),
            dict(session="s", event_index=0, model="b", log_evidence=4.5, status="success", evidence_comparable=True),
        ]
    )

    row = diagnostics.evidence_margin_table(scores).iloc[0]

    assert row["best_model_by_evidence"] == "a"
    assert row["second_best_model_by_evidence"] == "b"
    assert row["evidence_margin_to_second_best"] == pytest.approx(0.5)
    assert row["models_compared"] == 2
