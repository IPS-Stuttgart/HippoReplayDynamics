from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.advanced_result_diagnostics import (
    paired_model_margin_threshold_sweep,
    select_paired_model_margin_threshold,
)


def test_paired_threshold_sweep_preserves_schema_without_complete_pairs():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 1],
            "true_model": ["momentum", "diffusion"],
            "model": ["momentum", "momentum"],
            "log_evidence": [10.0, 8.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )

    sweep = paired_model_margin_threshold_sweep(
        scores,
        positive_model="momentum",
        reference_model="diffusion",
        thresholds=(0.0, 5.0),
        group_cols=("session", "event_index"),
        true_model_col="true_model",
        positive_true_label="momentum",
    )

    assert sweep["margin_threshold"].tolist() == [0.0, 5.0]
    assert sweep["positive_model"].tolist() == ["momentum", "momentum"]
    assert sweep["reference_model"].tolist() == ["diffusion", "diffusion"]
    assert sweep["events"].tolist() == [0, 0]
    assert sweep["false_positive_claims"].tolist() == [0, 0]
    assert sweep["positive_true_events"].tolist() == [0, 0]
    assert np.isnan(sweep["positive_claim_recall"]).all()

    selected = select_paired_model_margin_threshold(sweep, max_false_positive_claims=0)

    assert selected.loc[0, "selection_status"] == "fallback_no_gate_pass"
    assert selected.loc[0, "selected_margin_threshold"] == 0.0


def test_paired_threshold_sweep_rejects_malformed_thresholds():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["momentum", "diffusion"],
            "log_evidence": [10.0, 8.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )

    for thresholds in ((np.nan,), (np.inf,), (-1.0,), (True,), (False,), (np.bool_(True),)):
        with pytest.raises(ValueError, match="finite nonnegative"):
            paired_model_margin_threshold_sweep(
                scores,
                positive_model="momentum",
                reference_model="diffusion",
                thresholds=thresholds,
                group_cols=("session", "event_index"),
            )
