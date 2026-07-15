from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from candidate_support_convergence import infer_run_label  # noqa: E402


@pytest.mark.parametrize("invalid", ["not-a-count", True, np.inf])
def test_infer_run_label_rejects_malformed_candidate_counts(invalid: object) -> None:
    frame = pd.DataFrame(
        {
            "diagnostic_state_space_momentum_candidate_top_k": [64, invalid],
        }
    )

    with pytest.raises(ValueError, match="candidate counts"):
        infer_run_label(frame, "fallback")


def test_infer_run_label_ignores_only_explicit_missing_candidate_values() -> None:
    frame = pd.DataFrame(
        {
            "diagnostic_state_space_momentum_candidate_top_k": [64, "", pd.NA],
            "diagnostic_state_space_momentum_predicted_candidate_top_k": [
                8,
                "none",
                np.nan,
            ],
        }
    )

    assert infer_run_label(frame, "fallback") == "top_k=64,pred_k=8"
