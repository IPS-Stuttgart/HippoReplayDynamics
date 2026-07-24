from __future__ import annotations

import pandas as pd

from hipporeplayimm.sign_flip_report import score_table_sign_flip_summary


def test_sign_flip_preserves_distinct_invalid_utf8_model_labels() -> None:
    scores = pd.DataFrame(
        {
            "model": pd.Series([b"\xff", b"\xfe", "\ufffd"], dtype=object),
            "delta_vs_best_static": [1.0, -1.0, 2.0],
        }
    )

    summary = score_table_sign_flip_summary(scores)

    assert summary["model"].tolist() == [
        "<invalid-utf8-bytes:ff>",
        "<invalid-utf8-bytes:fe>",
        "\ufffd",
    ]
    assert summary["n_observations"].tolist() == [1, 1, 1]
    assert summary["permutations_evaluated"].tolist() == [2, 2, 2]
