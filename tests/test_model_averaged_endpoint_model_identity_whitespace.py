from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvement_extensions import add_model_averaged_endpoint_columns


def test_model_averaged_endpoint_deduplicates_padded_text_model_labels() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0] * 4,
            "model": ["diffusion", np.str_(" diffusion "), b"diffusion ", "momentum"],
            "evidence_comparable": [True] * 4,
            "model_probability": [0.2, 0.4, 0.8, 0.5],
            "log_evidence": [0.0, 1.0, 2.0, 1.0],
            "diagnostic_decoded_endpoint_x": [0.0, 50.0, 100.0, 20.0],
            "diagnostic_decoded_endpoint_y": [0.0, 25.0, 50.0, 10.0],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    assert out["model_averaged_endpoint_x"].tolist() == pytest.approx([90.0 / 1.3] * 4)
    assert out["model_averaged_endpoint_y"].tolist() == pytest.approx([45.0 / 1.3] * 4)
    assert out["model_averaged_endpoint_models"].tolist() == [2] * 4
    assert out["model_log_evidence_margin"].tolist() == pytest.approx([1.0] * 4)
