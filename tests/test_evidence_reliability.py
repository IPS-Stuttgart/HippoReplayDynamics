from __future__ import annotations

import pandas as pd

from hipporeplayimm.evidence_reliability import add_event_reliability_flags


def test_add_event_reliability_flags_replaces_existing_columns_on_rerun():
    scores = pd.DataFrame(
        [
            {
                "model": "diffusion",
                "status": "success",
                "n_spikes": 4,
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
            }
        ]
    )

    once = add_event_reliability_flags(scores)
    assert list(once.columns).count("event_reliable") == 1

    # Simulate rerunning the augmentation on an already augmented CSV.  The
    # stale flag value must be replaced rather than kept beside a duplicate
    # event_reliable column, otherwise downstream pandas aggregations can fail
    # or read the wrong column shape.
    once["event_reliable"] = False
    twice = add_event_reliability_flags(once)

    assert list(twice.columns).count("event_reliable") == 1
    assert bool(twice.loc[0, "event_reliable"])

    reliability = twice.groupby("model", as_index=False).agg(
        reliable_rows=("event_reliable", "sum"),
    )
    assert int(reliability.loc[0, "reliable_rows"]) == 1
