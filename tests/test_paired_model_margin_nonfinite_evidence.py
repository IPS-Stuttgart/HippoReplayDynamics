from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import paired_model_margin_decisions


def test_paired_model_margin_ignores_nonfinite_duplicate_evidence() -> None:
    momentum_model = "sorted-spike-state-space-momentum-exact-sparse"
    diffusion_model = "sorted-spike-state-space-diffusion"
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0, 0],
            "model": [momentum_model, momentum_model, diffusion_model, diffusion_model],
            "log_evidence": [np.inf, "10.0", -np.inf, "8.0"],
            "status": ["success", "success", "success", "success"],
            "evidence_comparable": [True, True, True, True],
        }
    )

    decisions = paired_model_margin_decisions(
        scores,
        positive_model=momentum_model,
        reference_model=diffusion_model,
        margin_threshold=0.0,
    )

    assert len(decisions) == 1
    assert decisions.loc[0, "positive_log_evidence"] == 10.0
    assert decisions.loc[0, "reference_log_evidence"] == 8.0
    assert decisions.loc[0, "positive_minus_reference_log_evidence"] == 2.0
    assert decisions.loc[0, "margin_decision"] == momentum_model
