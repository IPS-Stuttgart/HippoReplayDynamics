from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.model_averaged_endpoint_scoping import (
    add_model_averaged_endpoint_columns,
)


def test_evidence_margin_keeps_comparator_without_decoded_endpoint() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["winner", "comparator"],
            "evidence_comparable": [True, True],
            "model_probability": [0.8, 0.2],
            "log_evidence": [5.0, 1.0],
            "diagnostic_decoded_endpoint_x": [3.0, np.nan],
            "diagnostic_decoded_endpoint_y": [4.0, np.nan],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    np.testing.assert_allclose(out["model_averaged_endpoint_x"], 3.0)
    np.testing.assert_allclose(out["model_averaged_endpoint_y"], 4.0)
    assert out["model_averaged_endpoint_models"].tolist() == [1, 1]
    np.testing.assert_allclose(out["model_log_evidence_margin"], 4.0)
