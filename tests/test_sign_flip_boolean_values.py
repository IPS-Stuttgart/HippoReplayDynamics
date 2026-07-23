from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.sign_flip_report import paired_sign_flip_test, score_table_sign_flip_summary


@pytest.mark.parametrize(
    "values",
    [
        [1.0, True, -1.0],
        np.array([1.0, np.bool_(False), -1.0], dtype=object),
        pd.Series([1.0, True, -1.0], dtype=object),
    ],
)
def test_paired_sign_flip_rejects_booleans_mixed_with_numeric_deltas(values) -> None:
    with pytest.raises(ValueError, match="not booleans"):
        paired_sign_flip_test(values)


def test_score_table_sign_flip_summary_rejects_boolean_deltas() -> None:
    frame = pd.DataFrame(
        {
            "model": ["imm", "imm", "imm"],
            "delta_vs_best_static": pd.Series([1.0, True, -1.0], dtype=object),
        }
    )

    with pytest.raises(ValueError, match="boolean values"):
        score_table_sign_flip_summary(frame)
