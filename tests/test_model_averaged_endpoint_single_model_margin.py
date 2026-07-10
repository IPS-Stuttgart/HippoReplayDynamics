from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.model_averaged_endpoint_scoping import (
    add_model_averaged_endpoint_columns,
)


def test_model_averaged_endpoint_single_finite_model_has_unknown_margin() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "model": ["state-space-first-order-imm"],
            "evidence_comparable": [True],
            "model_probability": [1.0],
            "log_evidence": [5.0],
            "diagnostic_decoded_endpoint_x": [3.0],
            "diagnostic_decoded_endpoint_y": [4.0],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    assert out.loc[0, "model_averaged_endpoint_x"] == 3.0
    assert out.loc[0, "model_averaged_endpoint_y"] == 4.0
    assert out.loc[0, "model_averaged_endpoint_models"] == 1
    assert np.isnan(out.loc[0, "model_log_evidence_margin"])
