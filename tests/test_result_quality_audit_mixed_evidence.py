from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import ensure_evidence_support_columns
from hipporeplayimm.result_quality_audit import write_result_quality_audit


def test_result_quality_audit_keeps_mixed_evidence_rows(tmp_path) -> None:
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "exact-model",
                "status": "success",
                "evidence_support": "exact_full_grid",
                "log_evidence": 1.0,
                "heldout_log_likelihood": np.nan,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "heldout-model",
                "status": "success",
                "evidence_support": "exact_full_grid",
                "log_evidence": np.nan,
                "heldout_log_likelihood": 3.0,
            },
        ]
    )

    dashboard = write_result_quality_audit(scores, tmp_path)

    assert dashboard.exists()
    enriched = pd.read_csv(tmp_path / "event_model_evidence_with_quality.csv")
    assert enriched["log_evidence"].tolist() == [1.0, 3.0]
    assert enriched["evidence_comparable"].tolist() == [True, True]

    margins = pd.read_csv(tmp_path / "evidence_margins.csv")
    assert margins.loc[0, "best_model_by_evidence"] == "heldout-model"
    assert margins.loc[0, "second_best_model_by_evidence"] == "exact-model"
    assert margins.loc[0, "evidence_margin_to_second_best"] == 2.0


def test_mixed_evidence_finiteness_stays_conservative() -> None:
    scores = pd.DataFrame(
        {
            "log_evidence": [2.0, np.nan, np.nan, np.inf, "broken"],
            "heldout_log_likelihood": [np.nan, 3.0, np.nan, np.nan, 4.0],
        }
    )

    scored = ensure_evidence_support_columns(scores)

    assert scored["evidence_comparable"].tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
