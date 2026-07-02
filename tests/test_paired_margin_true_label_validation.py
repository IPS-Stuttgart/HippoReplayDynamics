from __future__ import annotations

import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import advanced_result_diagnostics as diagnostics


def test_paired_margin_decisions_reject_mixed_true_labels_within_scope() -> None:
    hipporeplayimm.apply_runtime_patches()
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0] * 4,
            "model": ["momentum", "diffusion", "momentum", "diffusion"],
            "log_evidence": [2.0, 1.0, 3.0, 0.5],
            "status": ["success"] * 4,
            "evidence_comparable": [True] * 4,
            "true_model": ["momentum", "momentum", "diffusion", "diffusion"],
        }
    )

    with pytest.raises(ValueError, match="true_model.*constant"):
        diagnostics.paired_model_margin_decisions(
            scores,
            positive_model="momentum",
            reference_model="diffusion",
            true_model_col="true_model",
        )
