from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvement_extensions import add_model_averaged_endpoint_columns


def _row(
    *,
    variant: str,
    window_index: int,
    model: str,
    probability: float,
    endpoint_x: float,
    endpoint_y: float = 0.0,
    log_evidence: float = 0.0,
) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": 7,
        "window_index": int(window_index),
        "event_window_variant": str(variant),
        "window_start_s": 10.0 + float(window_index),
        "window_end_s": 10.1 + float(window_index),
        "window_duration_s": 0.1,
        "model": str(model),
        "log_evidence": float(log_evidence),
        "model_probability": float(probability),
        "evidence_comparable": True,
        "diagnostic_decoded_endpoint_x": float(endpoint_x),
        "diagnostic_decoded_endpoint_y": float(endpoint_y),
    }


def test_model_averaged_endpoint_scopes_replay_window_variants() -> None:
    frame = pd.DataFrame(
        [
            _row(variant="core", window_index=0, model="a", probability=0.25, endpoint_x=10.0, log_evidence=0.0),
            _row(variant="core", window_index=0, model="b", probability=0.75, endpoint_x=20.0, log_evidence=1.0),
            _row(variant="expanded", window_index=1, model="a", probability=0.50, endpoint_x=100.0, log_evidence=0.0),
            _row(variant="expanded", window_index=1, model="b", probability=0.50, endpoint_x=200.0, log_evidence=0.0),
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    core = out[out["event_window_variant"].eq("core")]
    expanded = out[out["event_window_variant"].eq("expanded")]
    np.testing.assert_allclose(core["model_averaged_endpoint_x"], 17.5)
    np.testing.assert_allclose(expanded["model_averaged_endpoint_x"], 150.0)
    assert core["model_averaged_endpoint_models"].tolist() == [2, 2]
    assert expanded["model_averaged_endpoint_models"].tolist() == [2, 2]
