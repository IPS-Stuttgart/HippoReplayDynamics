from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.model_averaged_endpoint_scoping import add_model_averaged_endpoint_columns


def test_model_averaged_endpoint_margin_is_nan_when_log_evidence_is_all_nonfinite():
    scores = pd.DataFrame(
        {
            "session": ["RatX/Open1", "RatX/Open1"],
            "event_index": [0, 0],
            "model": ["state-space-diffusion", "state-space-first-order-imm"],
            "evidence_comparable": [True, True],
            "model_probability": [0.25, 0.75],
            "log_evidence": [np.nan, np.nan],
            "diagnostic_decoded_endpoint_x": [0.0, 4.0],
            "diagnostic_decoded_endpoint_y": [1.0, 5.0],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    assert out["model_averaged_endpoint_x"].iloc[0] == pytest.approx(3.0)
    assert out["model_averaged_endpoint_y"].iloc[0] == pytest.approx(4.0)
    assert out["model_averaged_endpoint_models"].iloc[0] == 2
    assert out["model_log_evidence_margin"].isna().all()
