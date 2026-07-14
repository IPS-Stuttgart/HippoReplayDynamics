from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_quality_gates import model_quality_summary


def test_model_quality_summary_retains_missing_model_rows_without_label_collision() -> None:
    sentinel_like_model = "__hipporeplayimm_missing_model__"
    scores = pd.DataFrame(
        {
            "session": ["s0", "s0", "s0", "s0"],
            "event_index": [0, 1, 2, 3],
            "model": ["known", None, np.nan, sentinel_like_model],
            "log_evidence": [4.0, 3.0, 2.0, 1.0],
            "status": ["success", "success", "success", "success"],
            "evidence_support": ["exact_full_grid"] * 4,
            "evidence_comparable": [True] * 4,
            "runtime_s": [1.0, 2.0, 4.0, 8.0],
        }
    )

    summary = model_quality_summary(scores)

    assert len(summary) == 3
    missing = summary[summary["model"].isna()]
    assert len(missing) == 1
    assert int(missing.iloc[0]["rows"]) == 2
    assert int(missing.iloc[0]["successful_rows"]) == 2
    assert int(missing.iloc[0]["exact_comparable_rows"]) == 2
    assert float(missing.iloc[0]["mean_runtime_s"]) == pytest.approx(3.0)

    sentinel_row = summary[summary["model"].eq(sentinel_like_model)]
    assert len(sentinel_row) == 1
    assert int(sentinel_row.iloc[0]["rows"]) == 1
