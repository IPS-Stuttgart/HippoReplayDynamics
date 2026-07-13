from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvement_extensions import add_model_averaged_endpoint_columns


def _score_row(
    *,
    seed_column: str,
    seed: int,
    model: str,
    log_evidence: float,
    probability: float,
    endpoint_x: float,
    endpoint_y: float,
) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": 7,
        "window_index": 0,
        "event_window_variant": "core",
        seed_column: seed,
        "model": model,
        "log_evidence": log_evidence,
        "model_probability": probability,
        "evidence_comparable": True,
        "diagnostic_decoded_endpoint_x": endpoint_x,
        "diagnostic_decoded_endpoint_y": endpoint_y,
    }


@pytest.mark.parametrize("seed_column", ["random_seed", "null_random_seed"])
def test_model_averaged_endpoint_scopes_stochastic_seed_columns(seed_column: str) -> None:
    frame = pd.DataFrame(
        [
            _score_row(
                seed_column=seed_column,
                seed=1,
                model="a",
                log_evidence=0.0,
                probability=0.25,
                endpoint_x=0.0,
                endpoint_y=10.0,
            ),
            _score_row(
                seed_column=seed_column,
                seed=1,
                model="b",
                log_evidence=1.0,
                probability=0.75,
                endpoint_x=4.0,
                endpoint_y=14.0,
            ),
            _score_row(
                seed_column=seed_column,
                seed=2,
                model="a",
                log_evidence=0.0,
                probability=0.50,
                endpoint_x=100.0,
                endpoint_y=20.0,
            ),
            _score_row(
                seed_column=seed_column,
                seed=2,
                model="b",
                log_evidence=0.0,
                probability=0.50,
                endpoint_x=200.0,
                endpoint_y=40.0,
            ),
        ]
    )

    out = add_model_averaged_endpoint_columns(frame)

    seed_one = out[out[seed_column].eq(1)]
    seed_two = out[out[seed_column].eq(2)]
    np.testing.assert_allclose(seed_one["model_averaged_endpoint_x"], 3.0)
    np.testing.assert_allclose(seed_one["model_averaged_endpoint_y"], 13.0)
    np.testing.assert_allclose(seed_one["model_log_evidence_margin"], 1.0)
    np.testing.assert_allclose(seed_two["model_averaged_endpoint_x"], 150.0)
    np.testing.assert_allclose(seed_two["model_averaged_endpoint_y"], 30.0)
    np.testing.assert_allclose(seed_two["model_log_evidence_margin"], 0.0)
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2, 2, 2]
