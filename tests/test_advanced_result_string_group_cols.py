from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    paired_model_margin_decisions,
    paired_model_margin_threshold_sweep,
)


def test_paired_model_margin_decisions_accepts_single_group_column_string() -> None:
    decisions = paired_model_margin_decisions(
        _paired_scores(),
        positive_model="momentum",
        reference_model="diffusion",
        group_cols="session",
    )

    assert decisions["session"].tolist() == ["Rat1/Open1"]
    assert decisions["positive_minus_reference_log_evidence"].tolist() == [2.0]
    assert decisions["positive_model_claimed"].tolist() == [True]


def test_paired_model_margin_threshold_sweep_accepts_single_group_column_string() -> None:
    sweep = paired_model_margin_threshold_sweep(
        _paired_scores(),
        positive_model="momentum",
        reference_model="diffusion",
        thresholds=(0.0, 1.5),
        group_cols="session",
    )

    assert sweep["events"].tolist() == [1, 1]
    assert sweep["group_cols"].tolist() == ["session", "session"]
    assert sweep["margin_threshold"].tolist() == [0.0, 1.5]


def _paired_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "model": ["momentum", "diffusion"],
            "log_evidence": [2.0, 0.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )
