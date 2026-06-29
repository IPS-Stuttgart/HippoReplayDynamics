from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.result_improvements import posterior_calibration_summary


def test_posterior_calibration_summary_ignores_impossible_rank_fractions() -> None:
    samples = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 5,
            "true_bin_probability": [0.25, 0.25, 0.25, 0.25, 0.25],
            "true_bin_rank": [1, 0, 12, 2.5, -1],
            "n_position_bins": [10, 10, 10, 10, 10],
        }
    )

    summary = posterior_calibration_summary(samples)

    assert summary.loc[0, "rows"] == 5
    assert summary.loc[0, "median_rank_fraction"] == pytest.approx(0.1)
    assert summary.loc[0, "coverage_50_rank"] == pytest.approx(1.0)
