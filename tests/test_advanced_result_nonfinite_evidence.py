from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    add_evidence_margin_columns,
    evidence_margin_table,
    paired_model_margin_decisions,
)


def test_evidence_margin_table_ignores_nonfinite_evidence_rows() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0, 0],
            "model": ["stationary", "diffusion", "momentum", "imm"],
            "log_evidence": [np.inf, "7.0", 2.0, -np.inf],
            "status": ["success", "success", "success", "success"],
            "evidence_comparable": [True, True, True, True],
        }
    )

    margins = evidence_margin_table(scores)

    assert len(margins) == 1
    assert margins.loc[0, "best_model_by_evidence"] == "diffusion"
    assert margins.loc[0, "second_best_model_by_evidence"] == "momentum"
    assert np.isclose(margins.loc[0, "best_log_evidence"], 7.0)
    assert np.isclose(margins.loc[0, "second_best_log_evidence"], 2.0)
    assert np.isclose(margins.loc[0, "evidence_margin_to_second_best"], 5.0)
    assert margins.loc[0, "models_compared"] == 2


def test_evidence_margin_columns_mark_all_nonfinite_groups_missing() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "diffusion"],
            "log_evidence": [np.inf, -np.inf],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )

    annotated = add_evidence_margin_columns(scores)

    assert annotated["evidence_margin_category"].tolist() == ["missing", "missing"]
    assert annotated["evidence_margin_to_second_best"].isna().all()


def test_paired_margin_decisions_skip_nonfinite_evidence_pairs() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 1, 1],
            "model": ["stationary", "diffusion", "stationary", "diffusion"],
            "log_evidence": [1.0, np.inf, 1.0, "4.0"],
            "status": ["success", "success", "success", "success"],
            "evidence_comparable": [True, True, True, True],
        }
    )

    decisions = paired_model_margin_decisions(
        scores,
        positive_model="diffusion",
        reference_model="stationary",
    )

    assert decisions["event_index"].tolist() == [1]
    assert np.isclose(decisions.loc[0, "positive_log_evidence"], 4.0)
    assert np.isclose(decisions.loc[0, "reference_log_evidence"], 1.0)
    assert np.isclose(decisions.loc[0, "positive_minus_reference_log_evidence"], 3.0)
    assert decisions.loc[0, "margin_decision"] == "diffusion"
