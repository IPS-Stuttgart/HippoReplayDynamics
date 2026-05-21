from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvement_extensions import add_model_averaged_endpoint_columns


def test_model_averaged_endpoint_accepts_legacy_tables_without_comparability_column() -> None:
    frame = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "a",
                "log_evidence": 0.0,
                "model_probability": 0.25,
                "diagnostic_decoded_endpoint_x": 10.0,
                "diagnostic_decoded_endpoint_y": 0.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "b",
                "log_evidence": 1.0,
                "model_probability": 0.75,
                "diagnostic_decoded_endpoint_x": 20.0,
                "diagnostic_decoded_endpoint_y": 10.0,
            },
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    assert np.allclose(out["model_averaged_endpoint_x"], 17.5)
    assert np.allclose(out["model_averaged_endpoint_y"], 7.5)
    assert (out["model_averaged_endpoint_models"] == 2).all()


def test_model_averaged_endpoint_uses_only_comparable_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "exact",
                "log_evidence": 0.0,
                "model_probability": 1.0,
                "evidence_comparable": True,
                "diagnostic_decoded_endpoint_x": 3.0,
                "diagnostic_decoded_endpoint_y": 4.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "truncated",
                "log_evidence": 100.0,
                "model_probability": 1.0,
                "evidence_comparable": False,
                "diagnostic_decoded_endpoint_x": 999.0,
                "diagnostic_decoded_endpoint_y": 999.0,
            },
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    assert np.allclose(out["model_averaged_endpoint_x"], 3.0)
    assert np.allclose(out["model_averaged_endpoint_y"], 4.0)
    assert (out["model_averaged_endpoint_models"] == 1).all()
