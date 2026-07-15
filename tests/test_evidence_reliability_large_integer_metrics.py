from __future__ import annotations

import pandas as pd

from hipporeplayimm.evidence_reliability import add_event_reliability_flags


def test_reliability_flags_keep_arbitrary_size_integer_counts_valid() -> None:
    huge_count = 10**400
    scores = pd.DataFrame(
        [
            {
                "model": "diffusion",
                "status": "success",
                "n_spikes": huge_count,
                "n_time": huge_count,
                "mean_candidate_log_mass": 0.0,
            }
        ]
    )

    flagged = add_event_reliability_flags(scores)

    assert bool(flagged.loc[0, "event_reliable"])
    assert not bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert not bool(flagged.loc[0, "event_low_spike_count"])
    assert not bool(flagged.loc[0, "event_too_few_time_bins"])
    assert flagged.loc[0, "event_reliability_reasons"] == ""
