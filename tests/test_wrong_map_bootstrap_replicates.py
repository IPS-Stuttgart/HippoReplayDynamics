from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.advanced_result_diagnostics import (
    rat_bootstrap_wrong_map_absolute_evidence_summary,
)


def _wrong_map_deltas() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat2/Open1"],
            "event_index": [0, 1],
            "statistic": ["best_exact_trajectory_model_real_map"] * 2,
            "statistic_type": ["real_map_selected_model"] * 2,
            "selected_model": ["m", "m"],
            "delta_map_log_evidence": [1.0, 2.0],
            "map_session": ["Rat1/Open2", "Rat2/Open2"],
        }
    )


@pytest.mark.parametrize("invalid", [0, -1, 1.5, np.nan, True, np.array([3])])
def test_wrong_map_bootstrap_rejects_invalid_replicate_counts(invalid: object) -> None:
    with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
        rat_bootstrap_wrong_map_absolute_evidence_summary(
            _wrong_map_deltas(),
            n_bootstrap=invalid,
        )


@pytest.mark.parametrize("invalid", [-1, 1.5, np.nan, True, np.array([7])])
def test_wrong_map_bootstrap_rejects_invalid_random_seed(invalid: object) -> None:
    with pytest.raises(ValueError, match="random_seed must be a finite nonnegative integer"):
        rat_bootstrap_wrong_map_absolute_evidence_summary(
            _wrong_map_deltas(),
            n_bootstrap=3,
            random_seed=invalid,
        )


def test_wrong_map_bootstrap_accepts_numeric_string_replicate_count() -> None:
    summary = rat_bootstrap_wrong_map_absolute_evidence_summary(
        _wrong_map_deltas(),
        n_bootstrap="3",
        random_seed=7,
    )

    assert summary.loc[0, "bootstrap_replicates"] == 3
    assert summary.loc[0, "observed_events"] == 2
    assert summary.loc[0, "observed_rats"] == 2


def test_wrong_map_bootstrap_filters_nonfinite_delta_rows() -> None:
    deltas = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat3/Open1"],
            "event_index": [0, 1, 2, 3],
            "statistic": ["fixed_model"] * 4,
            "statistic_type": ["fixed_model"] * 4,
            "selected_model": ["m", "m", "m", "m"],
            "delta_map_log_evidence": [1.0, np.nan, "not-a-number", 3.0],
            "map_session": ["Rat1/Open2", "Rat1/Open1", "Rat2/Open2", "Rat3/Open2"],
        }
    )

    summary = rat_bootstrap_wrong_map_absolute_evidence_summary(
        deltas,
        n_bootstrap=4,
        random_seed=3,
    )

    assert summary.loc[0, "observed_events"] == 2
    assert summary.loc[0, "observed_rats"] == 2
    assert summary.loc[0, "observed_positive_delta_fraction"] == 1.0
    assert summary.loc[0, "observed_mean_delta_map_log_evidence"] == 2.0
    assert np.isfinite(summary.loc[0, "mean_delta_ci95_low"])
    assert np.isfinite(summary.loc[0, "median_delta_ci95_high"])