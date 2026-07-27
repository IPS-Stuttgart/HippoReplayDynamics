from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvements import posterior_calibration_summary


def test_posterior_calibration_ignores_boolean_rank_metadata() -> None:
    samples = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "model": ["diffusion"] * 4,
            "true_bin_probability": [0.25, 0.5, 0.75, 1.0],
            "true_bin_rank": [True, np.bool_(True), np.array(True), 2],
            "n_position_bins": [4, np.array(4), True, 4],
        }
    )

    summary = posterior_calibration_summary(samples)

    assert summary.loc[0, "rows"] == 4
    assert summary.loc[0, "mean_true_probability"] == pytest.approx(0.625)
    assert summary.loc[0, "median_rank_fraction"] == pytest.approx(0.5)
    assert summary.loc[0, "coverage_50_rank"] == pytest.approx(1.0)
    assert summary.loc[0, "coverage_80_rank"] == pytest.approx(1.0)
    assert summary.loc[0, "coverage_95_rank"] == pytest.approx(1.0)


def test_posterior_calibration_returns_missing_rank_metrics_for_boolean_pairs() -> None:
    samples = pd.DataFrame(
        {
            "true_bin_probability": [0.25, 0.75],
            "true_bin_rank": [True, np.array(False)],
            "n_position_bins": [np.bool_(True), 4],
        }
    )

    summary = posterior_calibration_summary(samples)

    assert summary.loc[0, "rows"] == 2
    assert np.isnan(summary.loc[0, "median_rank_fraction"])
    assert np.isnan(summary.loc[0, "coverage_50_rank"])
    assert np.isnan(summary.loc[0, "coverage_80_rank"])
    assert np.isnan(summary.loc[0, "coverage_95_rank"])
