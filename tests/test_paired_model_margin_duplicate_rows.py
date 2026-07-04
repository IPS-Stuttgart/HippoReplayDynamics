from __future__ import annotations

import pandas as pd

import hipporeplayimm
from hipporeplayimm import advanced_result_diagnostics as diagnostics
from hipporeplayimm.advanced_result_diagnostics import paired_model_margin_decisions


def _duplicate_scores() -> pd.DataFrame:
    momentum_model = "sorted-spike-state-space-momentum-exact-sparse"
    diffusion_model = "sorted-spike-state-space-diffusion"
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0],
            "model": [momentum_model, diffusion_model, momentum_model],
            "log_evidence": ["10.0", "8.0", "7.0"],
            "status": ["success", "success", "success"],
            "evidence_comparable": [True, True, True],
        }
    )


def _assert_uses_best_duplicate_evidence(decisions: pd.DataFrame) -> None:
    momentum_model = "sorted-spike-state-space-momentum-exact-sparse"
    assert decisions.loc[0, "positive_log_evidence"] == 10.0
    assert decisions.loc[0, "reference_log_evidence"] == 8.0
    assert decisions.loc[0, "positive_minus_reference_log_evidence"] == 2.0
    assert decisions.loc[0, "margin_decision"] == momentum_model
    assert bool(decisions.loc[0, "positive_model_claimed"])


def test_duplicate_model_rows_use_best_finite_evidence() -> None:
    momentum_model = "sorted-spike-state-space-momentum-exact-sparse"
    diffusion_model = "sorted-spike-state-space-diffusion"

    decisions = paired_model_margin_decisions(
        _duplicate_scores(),
        positive_model=momentum_model,
        reference_model=diffusion_model,
        margin_threshold=0.0,
    )

    _assert_uses_best_duplicate_evidence(decisions)


def test_duplicate_model_rows_use_best_finite_evidence_in_patched_base() -> None:
    momentum_model = "sorted-spike-state-space-momentum-exact-sparse"
    diffusion_model = "sorted-spike-state-space-diffusion"
    hipporeplayimm.apply_runtime_patches()
    base_decisions = getattr(
        diagnostics,
        "_advanced_result_threshold_validation_base_decisions",
    )

    decisions = base_decisions(
        _duplicate_scores(),
        positive_model=momentum_model,
        reference_model=diffusion_model,
        margin_threshold=0.0,
    )

    _assert_uses_best_duplicate_evidence(decisions)
