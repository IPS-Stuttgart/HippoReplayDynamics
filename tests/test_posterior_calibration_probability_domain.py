from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvements import posterior_calibration_summary


def test_posterior_calibration_summary_drops_out_of_range_probabilities() -> None:
    samples = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 5,
            "true_bin_probability": [0.25, 0.0, -0.10, 1.20, np.nan],
            "true_bin_rank": [1, 2, 3, 4, 5],
            "n_position_bins": [4, 4, 4, 4, 4],
        }
    )

    summary = posterior_calibration_summary(samples)

    assert summary.loc[0, "rows"] == 2
    assert summary.loc[0, "mean_true_probability"] == pytest.approx(
        float(np.mean([0.25, np.finfo(float).tiny]))
    )
    assert summary.loc[0, "median_rank_fraction"] == pytest.approx(0.375)
    assert summary.loc[0, "coverage_50_rank"] == pytest.approx(1.0)
