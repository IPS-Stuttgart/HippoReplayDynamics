from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.evidence_reliability import add_event_reliability_flags


def _score_row() -> dict[str, object]:
    return {
        "model": "diffusion",
        "status": "success",
        "n_spikes": 4,
        "n_time": 3,
        "mean_candidate_log_mass": 0.0,
    }


@pytest.mark.parametrize(
    ("column", "value", "threshold_flag"),
    [
        ("n_spikes", -1, "event_low_spike_count"),
        ("n_spikes", 3.5, "event_low_spike_count"),
        ("n_spikes", "3.5", "event_low_spike_count"),
        ("n_time", -1, "event_too_few_time_bins"),
        ("n_time", 2.5, "event_too_few_time_bins"),
        ("n_time", np.float64(2.5), "event_too_few_time_bins"),
    ],
)
def test_reliability_flags_reject_nonintegral_or_negative_count_metrics(
    column: str,
    value: object,
    threshold_flag: str,
) -> None:
    row = _score_row()
    row[column] = value

    flagged = add_event_reliability_flags(pd.DataFrame([row]))

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert not bool(flagged.loc[0, threshold_flag])
    assert flagged.loc[0, "event_reliability_reasons"] == "invalid_numeric_metric"


def test_reliability_flags_keep_integral_numeric_count_representations_valid() -> None:
    row = _score_row()
    row["n_spikes"] = "4.0"
    row["n_time"] = np.float64(3.0)

    flagged = add_event_reliability_flags(pd.DataFrame([row]))

    assert bool(flagged.loc[0, "event_reliable"])
    assert not bool(flagged.loc[0, "event_invalid_numeric_metric"])
