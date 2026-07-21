from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvement_extensions import (
    add_model_averaged_endpoint_columns,
)


def _row(
    event_id: str,
    model: str,
    probability: float,
    log_evidence: float,
    endpoint_x: float,
) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": 0,
        "event_id": event_id,
        "model": model,
        "model_probability": probability,
        "log_evidence": log_evidence,
        "evidence_comparable": True,
        "diagnostic_decoded_endpoint_x": endpoint_x,
        "diagnostic_decoded_endpoint_y": 0.0,
    }


def test_model_averaged_endpoints_are_scoped_by_event_id() -> None:
    frame = pd.DataFrame(
        [
            _row("evt-a", "a", 0.25, 2.0, 0.0),
            _row("evt-a", "b", 0.75, 1.0, 4.0),
            _row("evt-b", "a", 0.50, 4.0, 100.0),
            _row("evt-b", "b", 0.50, 2.0, 200.0),
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    per_event = out.groupby("event_id", sort=True).first()
    assert per_event["model_averaged_endpoint_x"].to_dict() == {
        "evt-a": 3.0,
        "evt-b": 150.0,
    }
    assert per_event["model_averaged_endpoint_models"].to_dict() == {
        "evt-a": 2,
        "evt-b": 2,
    }
    assert per_event["model_log_evidence_margin"].to_dict() == {
        "evt-a": 1.0,
        "evt-b": 2.0,
    }
    np.testing.assert_allclose(
        per_event["model_averaged_endpoint_y"].to_numpy(dtype=float),
        0.0,
    )
