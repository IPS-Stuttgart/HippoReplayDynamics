from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.model_averaged_endpoint_scoping import add_model_averaged_endpoint_columns


def test_model_averaged_endpoint_margin_is_nan_when_log_evidence_is_all_nonfinite():
    scores = pd.DataFrame(
        {
            "session": ["RatX/Open1", "RatX/Open1"],
            "event_index": [0, 0],
            "model": ["state-space-diffusion", "state-space-first-order-imm"],
            "evidence_comparable": [True, True],
            "model_probability": [0.25, 0.75],
            "log_evidence": [np.nan, np.nan],
            "diagnostic_decoded_endpoint_x": [0.0, 4.0],
            "diagnostic_decoded_endpoint_y": [1.0, 5.0],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    assert out["model_averaged_endpoint_x"].iloc[0] == pytest.approx(3.0)
    assert out["model_averaged_endpoint_y"].iloc[0] == pytest.approx(4.0)
    assert out["model_averaged_endpoint_models"].iloc[0] == 2
    assert out["model_log_evidence_margin"].isna().all()


def test_model_averaged_endpoint_scopes_duplicate_index_by_row_position() -> None:
    scores = pd.DataFrame(
        {
            "session": ["RatX/Open1"] * 4,
            "event_index": [0, 0, 0, 0],
            "event_window_variant": ["core", "core", "expanded", "expanded"],
            "window_index": [0, 0, 1, 1],
            "model": ["diffusion", "first-order-imm", "diffusion", "first-order-imm"],
            "evidence_comparable": [True, True, True, True],
            "model_probability": [0.25, 0.75, 0.5, 0.5],
            "log_evidence": [0.0, 1.0, 0.0, 0.0],
            "diagnostic_decoded_endpoint_x": [0.0, 4.0, 100.0, 200.0],
            "diagnostic_decoded_endpoint_y": [10.0, 14.0, 20.0, 40.0],
        },
        index=[7, 7, 7, 7],
    )

    out = add_model_averaged_endpoint_columns(scores)

    core = out[out["event_window_variant"].eq("core")]
    expanded = out[out["event_window_variant"].eq("expanded")]
    assert core["model_averaged_endpoint_x"].tolist() == pytest.approx([3.0, 3.0])
    assert core["model_averaged_endpoint_y"].tolist() == pytest.approx([13.0, 13.0])
    assert expanded["model_averaged_endpoint_x"].tolist() == pytest.approx([150.0, 150.0])
    assert expanded["model_averaged_endpoint_y"].tolist() == pytest.approx([30.0, 30.0])
    assert core["model_averaged_endpoint_models"].tolist() == [2, 2]
    assert expanded["model_averaged_endpoint_models"].tolist() == [2, 2]


def test_model_averaged_endpoint_keeps_adjacent_large_event_ids_separate() -> None:
    lower = 2**53
    upper = lower + 1
    scores = pd.DataFrame(
        {
            "session": ["RatX/Open1"] * 4,
            "event_index": [lower, lower, upper, upper],
            "model": ["diffusion", "first-order-imm", "diffusion", "first-order-imm"],
            "evidence_comparable": [True] * 4,
            "model_probability": [0.5] * 4,
            "log_evidence": [0.0] * 4,
            "diagnostic_decoded_endpoint_x": [0.0, 2.0, 100.0, 200.0],
            "diagnostic_decoded_endpoint_y": [10.0, 14.0, 20.0, 40.0],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    lower_rows = out[out["event_index"].eq(lower)]
    upper_rows = out[out["event_index"].eq(upper)]
    assert lower_rows["model_averaged_endpoint_x"].tolist() == pytest.approx([1.0, 1.0])
    assert lower_rows["model_averaged_endpoint_y"].tolist() == pytest.approx([12.0, 12.0])
    assert upper_rows["model_averaged_endpoint_x"].tolist() == pytest.approx([150.0, 150.0])
    assert upper_rows["model_averaged_endpoint_y"].tolist() == pytest.approx([30.0, 30.0])


def test_model_averaged_endpoint_accepts_scope_integers_beyond_float_range() -> None:
    event_index = 10**400
    scores = pd.DataFrame(
        {
            "session": ["RatX/Open1", "RatX/Open1"],
            "event_index": [event_index, event_index],
            "model": ["diffusion", "first-order-imm"],
            "evidence_comparable": [True, True],
            "model_probability": [0.25, 0.75],
            "log_evidence": [0.0, 1.0],
            "diagnostic_decoded_endpoint_x": [0.0, 4.0],
            "diagnostic_decoded_endpoint_y": [10.0, 14.0],
        }
    )

    out = add_model_averaged_endpoint_columns(scores)

    assert out["model_averaged_endpoint_x"].tolist() == pytest.approx([3.0, 3.0])
    assert out["model_averaged_endpoint_y"].tolist() == pytest.approx([13.0, 13.0])
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2]
