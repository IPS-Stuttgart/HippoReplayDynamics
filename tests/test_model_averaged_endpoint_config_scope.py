from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.model_averaged_endpoint_scoping import add_model_averaged_endpoint_columns


def test_model_averaged_endpoint_scopes_model_configuration_metadata() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0] * 4,
            "window_index": [0] * 4,
            "model": ["diffusion", "first-order-imm", "diffusion", "first-order-imm"],
            "state_space_diffusion_sigma_cm_sqrt_s": [50.0, 50.0, 100.0, 100.0],
            "evidence_comparable": [True] * 4,
            "model_probability": [0.25, 0.75, 0.5, 0.5],
            "log_evidence": [0.0, 1.0, 0.0, 0.0],
            "diagnostic_decoded_endpoint_x": [0.0, 4.0, 100.0, 200.0],
            "diagnostic_decoded_endpoint_y": [10.0, 14.0, 20.0, 40.0],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    sigma_50 = out[out["state_space_diffusion_sigma_cm_sqrt_s"].eq(50.0)]
    sigma_100 = out[out["state_space_diffusion_sigma_cm_sqrt_s"].eq(100.0)]
    assert sigma_50["model_averaged_endpoint_x"].tolist() == pytest.approx([3.0, 3.0])
    assert sigma_50["model_averaged_endpoint_y"].tolist() == pytest.approx([13.0, 13.0])
    assert sigma_100["model_averaged_endpoint_x"].tolist() == pytest.approx([150.0, 150.0])
    assert sigma_100["model_averaged_endpoint_y"].tolist() == pytest.approx([30.0, 30.0])
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2, 2, 2]
