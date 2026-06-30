from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_quality_audit import write_result_quality_audit


def test_result_quality_audit_accepts_heldout_only_score_tables(tmp_path):
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-stationary",
                "status": "success",
                "evidence_support": "exact_full_grid",
                "heldout_log_likelihood": 1.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "status": "success",
                "evidence_support": "exact_full_grid",
                "heldout_log_likelihood": 3.0,
            },
        ]
    )

    dashboard = write_result_quality_audit(scores, tmp_path)

    assert dashboard.exists()
    enriched = pd.read_csv(tmp_path / "event_model_evidence_with_quality.csv")
    assert "log_evidence" in enriched.columns
    assert enriched["log_evidence"].tolist() == [1.0, 3.0]

    margins = pd.read_csv(tmp_path / "evidence_margins.csv")
    assert margins.loc[0, "best_model_by_evidence"] == "sorted-spike-state-space-diffusion"
    assert margins.loc[0, "second_best_model_by_evidence"] == "sorted-spike-state-space-stationary"
    assert margins.loc[0, "evidence_margin_to_second_best"] == 2.0
