from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    rat_bootstrap_wrong_map_absolute_evidence_summary,
)


def test_wrong_map_rat_bootstrap_uses_statistic_specific_rats():
    deltas = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "statistic": "statistic_present_in_rat1_only",
                "delta_map_log_evidence": 1.0,
                "selected_model": "sorted-spike-state-space-diffusion",
            },
            {
                "session": "Rat2/Open1",
                "statistic": "different_statistic_present_in_rat2_only",
                "delta_map_log_evidence": 2.0,
                "selected_model": "sorted-spike-state-space-fragmented",
            },
        ]
    )

    summary = rat_bootstrap_wrong_map_absolute_evidence_summary(
        deltas,
        n_bootstrap=16,
        random_seed=7,
    )

    row = summary[
        summary["statistic"].eq("statistic_present_in_rat1_only")
    ].iloc[0]

    assert row["observed_rats"] == 1
    assert row["observed_events"] == 1
    assert row["observed_mean_delta_map_log_evidence"] == 1.0
    assert row["mean_delta_ci95_low"] == 1.0
    assert row["mean_delta_ci95_high"] == 1.0
    assert row["observed_median_delta_map_log_evidence"] == 1.0
    assert row["median_delta_ci95_low"] == 1.0
    assert row["median_delta_ci95_high"] == 1.0
    assert np.isfinite(row["positive_delta_fraction_ci95_low"])
