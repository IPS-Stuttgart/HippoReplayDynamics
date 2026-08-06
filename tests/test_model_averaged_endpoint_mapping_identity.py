from __future__ import annotations

import math

import pandas as pd
import pytest

from hipporeplayimm.model_averaged_endpoint_scoping import (
    _model_identity,
    add_model_averaged_endpoint_columns,
)


def test_model_average_deduplicates_equivalent_mapping_labels() -> None:
    primary = {"family": "state-space", "mode": "imm"}
    reordered_primary = {"mode": "imm", "family": "state-space"}
    frame = pd.DataFrame(
        {
            "session": ["RatX/Open1"] * 3,
            "event_index": [7] * 3,
            "model": [primary, reordered_primary, "runner-up"],
            "model_probability": [0.6, 0.6, 0.4],
            "diagnostic_decoded_endpoint_x": [10.0, 100.0, 0.0],
            "diagnostic_decoded_endpoint_y": [20.0, 200.0, 0.0],
            "log_evidence": [9.0, 8.0, 3.0],
            "evidence_comparable": [True, True, True],
        }
    )

    result = add_model_averaged_endpoint_columns(frame)

    assert result["model_averaged_endpoint_models"].eq(2).all()
    assert result["model_averaged_endpoint_x"].eq(6.0).all()
    assert result["model_averaged_endpoint_y"].eq(12.0).all()
    assert result["model_log_evidence_margin"].eq(6.0).all()
    expected_entropy = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
    assert result["model_probability_entropy"].eq(
        pytest.approx(expected_entropy)
    ).all()


def test_model_identity_canonicalizes_unordered_sets() -> None:
    assert _model_identity({"diffusion", "stationary"}) == _model_identity(
        {"stationary", "diffusion"}
    )
