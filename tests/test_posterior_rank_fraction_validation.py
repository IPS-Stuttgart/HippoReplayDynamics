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


def test_posterior_calibration_summary_ignores_rank_below_one() -> None:
    samples = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 2,
            "true_bin_probability": [0.25, 0.25],
            "true_bin_rank": [0, 2],
            "n_position_bins": [10, 10],
        }
    )

    summary = posterior_calibration_summary(samples)

    assert summary.loc[0, "rows"] == 2
    assert summary.loc[0, "median_rank_fraction"] == pytest.approx(0.2)
    assert summary.loc[0, "coverage_50_rank"] == pytest.approx(1.0)


def test_posterior_calibration_summary_accepts_nullable_missing_values() -> None:
    samples = pd.DataFrame(
        {
            "session": pd.Series(["Rat1/Open1"] * 4, dtype="string"),
            "true_bin_probability": pd.Series([0.25, pd.NA, 0.50, 1.25], dtype="Float64"),
            "true_bin_rank": pd.Series([1, pd.NA, 5, 2], dtype="Int64"),
            "n_position_bins": pd.Series([10, 10, pd.NA, 10], dtype="Int64"),
        }
    )

    summary = posterior_calibration_summary(samples)

    assert summary.loc[0, "rows"] == 2
    assert summary.loc[0, "mean_true_probability"] == pytest.approx(0.375)
    assert summary.loc[0, "median_rank_fraction"] == pytest.approx(0.1)
    assert summary.loc[0, "coverage_50_rank"] == pytest.approx(1.0)
