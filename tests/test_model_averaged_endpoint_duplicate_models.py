from __future__ import annotations

import math

import pandas as pd
import pytest

from hipporeplayimm.model_averaged_endpoint_scoping import add_model_averaged_endpoint_columns


def test_model_averaged_endpoint_uses_one_row_per_model() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 3,
            "event_index": [0] * 3,
            "model": ["winner", "winner", "runner-up"],
            "evidence_comparable": [True] * 3,
            "model_probability": [0.55, 0.60, 0.40],
            "log_evidence": [9.0, 10.0, 4.0],
            "diagnostic_decoded_endpoint_x": [100.0, 0.0, 10.0],
            "diagnostic_decoded_endpoint_y": [200.0, 0.0, 20.0],
        },
        index=pd.Index([7, 7, 8], name="score_row"),
    )

    out = add_model_averaged_endpoint_columns(scores)

    assert out.index.equals(scores.index)
    assert out["model_averaged_endpoint_x"].tolist() == pytest.approx([4.0] * 3)
    assert out["model_averaged_endpoint_y"].tolist() == pytest.approx([8.0] * 3)
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2, 2]
    assert out["model_probability_entropy"].tolist() == pytest.approx(
        [-0.6 * math.log(0.6) - 0.4 * math.log(0.4)] * 3
    )
    assert out["model_log_evidence_margin"].tolist() == pytest.approx([6.0] * 3)
