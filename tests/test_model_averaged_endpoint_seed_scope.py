from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.model_averaged_endpoint_scoping import add_model_averaged_endpoint_columns


@pytest.mark.parametrize("seed_column", ["random_seed", "null_random_seed"])
def test_model_averaged_endpoints_are_scoped_by_stochastic_seed(seed_column: str) -> None:
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                seed_column: 1,
                "model": "diffusion",
                "evidence_comparable": True,
                "model_probability": 0.75,
                "log_evidence": 2.0,
                "diagnostic_decoded_endpoint_x": 0.0,
                "diagnostic_decoded_endpoint_y": 10.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                seed_column: 1,
                "model": "momentum",
                "evidence_comparable": True,
                "model_probability": 0.25,
                "log_evidence": 1.0,
                "diagnostic_decoded_endpoint_x": 4.0,
                "diagnostic_decoded_endpoint_y": 14.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                seed_column: 2,
                "model": "diffusion",
                "evidence_comparable": True,
                "model_probability": 0.25,
                "log_evidence": 0.0,
                "diagnostic_decoded_endpoint_x": 100.0,
                "diagnostic_decoded_endpoint_y": 20.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                seed_column: 2,
                "model": "momentum",
                "evidence_comparable": True,
                "model_probability": 0.75,
                "log_evidence": 2.0,
                "diagnostic_decoded_endpoint_x": 200.0,
                "diagnostic_decoded_endpoint_y": 40.0,
            },
        ]
    )

    out = add_model_averaged_endpoint_columns(scores)

    seed_one = out[out[seed_column].eq(1)]
    seed_two = out[out[seed_column].eq(2)]
    assert seed_one["model_averaged_endpoint_x"].tolist() == pytest.approx([1.0, 1.0])
    assert seed_one["model_averaged_endpoint_y"].tolist() == pytest.approx([11.0, 11.0])
    assert seed_one["model_log_evidence_margin"].tolist() == pytest.approx([1.0, 1.0])
    assert seed_two["model_averaged_endpoint_x"].tolist() == pytest.approx([175.0, 175.0])
    assert seed_two["model_averaged_endpoint_y"].tolist() == pytest.approx([35.0, 35.0])
    assert seed_two["model_log_evidence_margin"].tolist() == pytest.approx([2.0, 2.0])
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2, 2, 2]
