from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.sign_flip_report import (
    paired_sign_flip_test,
    score_table_sign_flip_summary,
)


@pytest.mark.parametrize(
    "values",
    [
        [1.0 + 2.0j, 2.0],
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
    ],
)
def test_paired_sign_flip_rejects_complex_deltas(values: object) -> None:
    with pytest.raises(ValueError, match="real numeric deltas, not complex values"):
        paired_sign_flip_test(values)


@pytest.mark.parametrize("value", [1.0 + 2.0j, np.complex128(1.0 + 0.0j)])
def test_score_table_sign_flip_rejects_complex_deltas(value: object) -> None:
    frame = pd.DataFrame(
        {
            "model": ["imm"],
            "delta_vs_best_static": [value],
        }
    )

    with pytest.raises(ValueError, match="delta_vs_best_static contains complex values"):
        score_table_sign_flip_summary(frame)
