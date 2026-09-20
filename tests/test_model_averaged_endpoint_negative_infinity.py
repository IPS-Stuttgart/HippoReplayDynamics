from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.model_averaged_endpoint_scoping import (
    _distinct_model_rows,
    add_model_averaged_endpoint_columns,
)


def test_model_averaged_endpoint_margin_preserves_negative_infinite_comparator() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["winner", "impossible"],
            "evidence_comparable": [True, True],
            "model_probability": [1.0, 0.0],
            "log_evidence": [5.0, -np.inf],
            "diagnostic_decoded_endpoint_x": [3.0, np.nan],
            "diagnostic_decoded_endpoint_y": [4.0, np.nan],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    np.testing.assert_allclose(out["model_averaged_endpoint_x"], 3.0)
    np.testing.assert_allclose(out["model_averaged_endpoint_y"], 4.0)
    assert out["model_averaged_endpoint_models"].tolist() == [1, 1]
    assert np.isposinf(out["model_log_evidence_margin"]).all()


def test_distinct_model_rows_prefers_negative_infinity_over_invalid_positive_infinity() -> None:
    frame = pd.DataFrame(
        {
            "model": ["impossible", "impossible", "winner"],
            "log_evidence": [np.inf, -np.inf, 0.0],
            "model_probability": [0.0, 0.0, 1.0],
            "diagnostic_decoded_endpoint_x": [999.0, 100.0, 3.0],
            "diagnostic_decoded_endpoint_y": [999.0, 200.0, 4.0],
        }
    )

    distinct = _distinct_model_rows(frame)

    impossible = distinct[distinct["model"].eq("impossible")].iloc[0]
    assert np.isneginf(impossible["log_evidence"])
