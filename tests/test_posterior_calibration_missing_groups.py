from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.result_improvements import posterior_calibration_summary


def test_posterior_calibration_summary_retains_missing_group_keys() -> None:
    samples = pd.DataFrame(
        {
            "session": ["Rat1/Open1", None, "Rat1/Open1"],
            "model": ["target", "target", None],
            "true_bin_probability": [0.25, 0.75, 0.5],
            "true_bin_rank": [1, 2, 1],
            "n_position_bins": [4, 4, 2],
        }
    )

    summary = posterior_calibration_summary(samples)

    assert int(summary["rows"].sum()) == 3

    missing_session = summary["session"].isna() & summary["model"].eq("target")
    assert int(missing_session.sum()) == 1
    assert summary.loc[missing_session, "mean_true_probability"].iloc[0] == pytest.approx(0.75)

    missing_model = summary["session"].eq("Rat1/Open1") & summary["model"].isna()
    assert int(missing_model.sum()) == 1
    assert summary.loc[missing_model, "mean_true_probability"].iloc[0] == pytest.approx(0.5)
