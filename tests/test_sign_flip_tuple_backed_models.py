from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.sign_flip_report import score_table_sign_flip_summary


def test_sign_flip_normalizes_singleton_sequence_model_labels() -> None:
    scores = pd.DataFrame(
        {
            "model": pd.Series(
                [
                    "imm",
                    [bytearray(b"imm")],
                    (bytearray(b"imm"),),
                    np.array([np.bytes_("imm")], dtype=object),
                ],
                dtype=object,
            ),
            "delta_vs_best_static": [1.0, 2.0, 3.0, 4.0],
        }
    )

    summary = score_table_sign_flip_summary(scores, models=["imm"])

    assert summary["model"].tolist() == ["imm"]
    assert summary["n_observations"].tolist() == [4]
    assert summary["permutations_evaluated"].tolist() == [16]


def test_sign_flip_recursively_normalizes_multi_element_tuples() -> None:
    scores = pd.DataFrame(
        {
            "model": pd.Series(
                [
                    (bytearray(b"imm"), np.bytes_("variant")),
                    [memoryview(b"imm"), "variant"],
                ],
                dtype=object,
            ),
            "delta_vs_best_static": [1.0, 2.0],
        }
    )

    summary = score_table_sign_flip_summary(scores)

    assert summary["model"].tolist() == [("imm", "variant")]
    assert summary["n_observations"].tolist() == [2]
