from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvements import summarize_grouped_model_metrics


def test_summarize_grouped_model_metrics_preserves_missing_group_values() -> None:
    rows = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "window_variant": ["core", np.nan],
            "model": ["state-space-first-order-imm", "state-space-first-order-imm"],
            "heldout_log_likelihood": [1.0, 2.0],
        }
    )

    summary = summarize_grouped_model_metrics(rows, ("session", "window_variant"))

    assert int(summary["events"].sum()) == 2
    missing = summary["window_variant"].isna()
    assert int(missing.sum()) == 1
    assert int(summary.loc[missing, "events"].iloc[0]) == 1
    assert summary.loc[missing, "mean_heldout_log_likelihood"].iloc[0] == pytest.approx(2.0)
