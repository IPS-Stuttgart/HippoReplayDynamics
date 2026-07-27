from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvements import posterior_calibration_summary


def test_posterior_calibration_drops_boolean_probability_rows() -> None:
    samples = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "model": ["diffusion"] * 4,
            "true_bin_probability": [
                True,
                np.bool_(False),
                np.array(True),
                0.25,
            ],
            "true_bin_rank": [1, 1, 1, 2],
            "n_position_bins": [4, 4, 4, 4],
        }
    )

    summary = posterior_calibration_summary(samples)

    assert summary.loc[0, "rows"] == 1
    assert summary.loc[0, "mean_true_probability"] == pytest.approx(0.25)
    assert summary.loc[0, "median_true_probability"] == pytest.approx(0.25)
    assert summary.loc[0, "mean_true_negative_log_probability"] == pytest.approx(
        -np.log(0.25)
    )
    assert summary.loc[0, "median_rank_fraction"] == pytest.approx(0.5)


def test_posterior_calibration_returns_empty_for_only_boolean_probabilities() -> None:
    samples = pd.DataFrame(
        {
            "true_bin_probability": [True, np.bool_(False), np.array(True)],
            "true_bin_rank": [1, 1, 1],
            "n_position_bins": [4, 4, 4],
        }
    )

    assert posterior_calibration_summary(samples).empty
