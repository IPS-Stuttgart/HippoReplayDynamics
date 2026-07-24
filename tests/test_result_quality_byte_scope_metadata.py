from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm import result_quality_audit as audit


def test_result_quality_ignores_byte_backed_missing_scope_metadata() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 3,
            "event_index": [0] * 3,
            "window_role": [b"NA", bytearray(b"none"), memoryview(b"null")],
            "model": ["stationary", "diffusion", "momentum"],
            "log_evidence": [3.0, 2.0, 1.0],
        }
    )

    assert audit.event_group_columns(scores) == ["session", "event_index"]
    assert audit._event_count(scores) == 1

    normalized = audit._score_table_with_log_evidence_alias(scores)
    assert normalized["window_role"].isna().all()


def test_result_quality_normalizes_hashable_scope_metadata() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0] * 4,
            "window_role": [
                b"core",
                bytearray(b"core"),
                memoryview(b"core"),
                np.bytes_("core"),
            ],
            "train_cell_ids": [
                np.array([1, 2]),
                [1, 2],
                (1, 2),
                np.array([1, 2]),
            ],
            "model": ["stationary", "diffusion", "momentum", "jump"],
            "log_evidence": [4.0, 3.0, 2.0, 1.0],
        }
    )

    normalized = audit._score_table_with_log_evidence_alias(scores)

    assert normalized["window_role"].tolist() == ["core"] * 4
    assert normalized["train_cell_ids"].tolist() == [("sequence", (1, 2))] * 4
    assert audit.event_group_columns(normalized) == [
        "session",
        "event_index",
        "window_role",
        "train_cell_ids",
    ]
    assert audit._event_count(scores) == 1
