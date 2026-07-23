from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.sign_flip_report import score_table_sign_flip_summary


def _byte_backed_model_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": pd.Series(
                [
                    b"imm",
                    bytearray(b"imm"),
                    memoryview(b"imm"),
                    np.bytes_(b"imm"),
                    "imm",
                ],
                dtype=object,
            ),
            "delta_vs_best_static": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_sign_flip_model_filter_decodes_byte_backed_labels() -> None:
    summary = score_table_sign_flip_summary(
        _byte_backed_model_rows(),
        models=["imm"],
    )

    assert summary["model"].tolist() == ["imm"]
    assert summary["n_observations"].tolist() == [5]
    assert summary["permutations_evaluated"].tolist() == [32]


def test_sign_flip_summary_groups_semantically_equal_model_labels() -> None:
    summary = score_table_sign_flip_summary(_byte_backed_model_rows())

    assert summary["model"].tolist() == ["imm"]
    assert summary["n_observations"].tolist() == [5]
