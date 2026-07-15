from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.model_averaged_endpoint_scoping import (
    add_model_averaged_endpoint_columns,
)


def test_model_averaged_endpoint_accepts_arbitrary_precision_comparable_flags() -> None:
    comparable = 10**400
    frame = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "diffusion"],
            "model_probability": [0.25, 0.75],
            "diagnostic_decoded_endpoint_x": [0.0, 8.0],
            "diagnostic_decoded_endpoint_y": [2.0, 6.0],
            "log_evidence": [0.0, 1.0],
            "evidence_comparable": pd.Series(
                [comparable, comparable],
                dtype=object,
            ),
        }
    )

    result = add_model_averaged_endpoint_columns(frame)

    assert result["model_averaged_endpoint_x"].tolist() == pytest.approx([6.0, 6.0])
    assert result["model_averaged_endpoint_y"].tolist() == pytest.approx([5.0, 5.0])
    assert result["model_averaged_endpoint_models"].tolist() == [2, 2]
    assert result["model_log_evidence_margin"].tolist() == pytest.approx([1.0, 1.0])
