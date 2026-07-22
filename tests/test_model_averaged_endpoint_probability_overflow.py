from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvement_extensions import (
    add_model_averaged_endpoint_columns,
)


def test_model_averaged_endpoints_normalize_large_finite_weights_stably() -> None:
    max_float = np.finfo(float).max
    frame = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "a",
                "model_probability": max_float,
                "log_evidence": 2.0,
                "evidence_comparable": True,
                "diagnostic_decoded_endpoint_x": 0.0,
                "diagnostic_decoded_endpoint_y": -2.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "b",
                "model_probability": max_float,
                "log_evidence": 1.0,
                "evidence_comparable": True,
                "diagnostic_decoded_endpoint_x": 2.0,
                "diagnostic_decoded_endpoint_y": 2.0,
            },
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    np.testing.assert_allclose(out["model_averaged_endpoint_x"], 1.0)
    np.testing.assert_allclose(out["model_averaged_endpoint_y"], 0.0)
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2]
    assert out["model_probability_entropy"].tolist() == pytest.approx(
        [np.log(2.0), np.log(2.0)]
    )
    assert out["model_log_evidence_margin"].tolist() == pytest.approx([1.0, 1.0])


def test_model_averaged_endpoints_average_large_finite_coordinates_stably() -> None:
    max_float = np.finfo(float).max
    frame = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "a",
                "model_probability": 0.9966546216081738,
                "log_evidence": 2.0,
                "evidence_comparable": True,
                "diagnostic_decoded_endpoint_x": max_float,
                "diagnostic_decoded_endpoint_y": 0.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "b",
                "model_probability": 0.003345378391826308,
                "log_evidence": 1.0,
                "evidence_comparable": True,
                "diagnostic_decoded_endpoint_x": max_float,
                "diagnostic_decoded_endpoint_y": 0.0,
            },
        ]
    )

    with np.errstate(over="raise", invalid="raise"):
        out = add_model_averaged_endpoint_columns(frame)

    assert out["model_averaged_endpoint_x"].tolist() == [max_float, max_float]
    assert out["model_averaged_endpoint_y"].tolist() == [0.0, 0.0]
