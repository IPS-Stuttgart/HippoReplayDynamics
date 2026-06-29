from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    add_evidence_margin_columns,
    evidence_margin_table,
)


def _scores_with_missing_window_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "event_window_variant": [pd.NA, pd.NA],
            "model": ["stationary", "diffusion"],
            "log_evidence": [10.0, 12.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )


def test_evidence_margin_table_keeps_missing_optional_group_metadata() -> None:
    scores = _scores_with_missing_window_metadata()

    margins = evidence_margin_table(
        scores,
        group_cols=("session", "event_index", "event_window_variant"),
    )

    assert len(margins) == 1
    assert margins["event_window_variant"].isna().all()
    assert margins.loc[0, "best_model_by_evidence"] == "diffusion"
    assert margins.loc[0, "second_best_model_by_evidence"] == "stationary"
    assert np.isclose(margins.loc[0, "evidence_margin_to_second_best"], 2.0)


def test_evidence_margin_columns_merge_back_missing_optional_group_metadata() -> None:
    scores = _scores_with_missing_window_metadata()

    merged = add_evidence_margin_columns(
        scores,
        group_cols=("session", "event_index", "event_window_variant"),
    )

    assert len(merged) == len(scores)
    assert merged["best_model_by_evidence"].tolist() == ["diffusion", "diffusion"]
    assert np.allclose(merged["evidence_margin_to_second_best"].to_numpy(dtype=float), [2.0, 2.0])
