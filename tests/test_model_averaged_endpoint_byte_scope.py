from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.model_averaged_endpoint_scoping import add_model_averaged_endpoint_columns


def _scores(*, session: list[object], variant: list[object], comparable: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": session,
            "event_index": [0, 0],
            "event_window_variant": variant,
            "model": ["diffusion", "first-order-imm"],
            "evidence_comparable": comparable,
            "model_probability": [0.25, 0.75],
            "log_evidence": [0.0, 1.0],
            "diagnostic_decoded_endpoint_x": [0.0, 4.0],
            "diagnostic_decoded_endpoint_y": [10.0, 14.0],
        }
    )


def test_model_averaged_endpoint_groups_byte_backed_scope_labels_with_text() -> None:
    scores = _scores(
        session=[np.bytes_("Rat1/Open1"), "Rat1/Open1"],
        variant=[bytearray(b"core"), "core"],
        comparable=[True, True],
    )

    out = add_model_averaged_endpoint_columns(scores)

    assert out["model_averaged_endpoint_x"].tolist() == pytest.approx([3.0, 3.0])
    assert out["model_averaged_endpoint_y"].tolist() == pytest.approx([13.0, 13.0])
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2]


def test_model_averaged_endpoint_accepts_byte_backed_true_flags() -> None:
    scores = _scores(
        session=["Rat1/Open1", "Rat1/Open1"],
        variant=["core", "core"],
        comparable=[np.bytes_("true"), memoryview(b"1")],
    )

    out = add_model_averaged_endpoint_columns(scores)

    assert out["model_averaged_endpoint_x"].tolist() == pytest.approx([3.0, 3.0])
    assert out["model_averaged_endpoint_y"].tolist() == pytest.approx([13.0, 13.0])
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2]
