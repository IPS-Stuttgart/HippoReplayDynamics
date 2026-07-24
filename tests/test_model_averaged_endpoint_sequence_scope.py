from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvement_extensions import add_model_averaged_endpoint_columns


def test_model_averaged_endpoint_merges_equivalent_sequence_scope_containers() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "train_cell_ids": [np.array([1, 2], dtype=np.int64), [1, 2]],
            "test_cell_ids": [np.array([3], dtype=np.int64), (3,)],
            "model": ["diffusion", "first-order-imm"],
            "evidence_comparable": [True, True],
            "model_probability": [0.25, 0.75],
            "log_evidence": [0.0, 1.0],
            "diagnostic_decoded_endpoint_x": [10.0, 20.0],
            "diagnostic_decoded_endpoint_y": [30.0, 50.0],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    assert out["model_averaged_endpoint_x"].tolist() == pytest.approx([17.5, 17.5])
    assert out["model_averaged_endpoint_y"].tolist() == pytest.approx([45.0, 45.0])
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2]
    assert out["model_log_evidence_margin"].tolist() == pytest.approx([1.0, 1.0])
