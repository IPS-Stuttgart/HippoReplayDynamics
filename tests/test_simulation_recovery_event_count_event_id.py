from __future__ import annotations

import pandas as pd

from hipporeplayimm.simulation_recovery_event_count import _distinct_event_count


def test_distinct_event_count_uses_event_id_without_event_index() -> None:
    events = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_id": [101, 102, 102],
            "true_model": ["momentum", "momentum", "momentum"],
        }
    )

    assert _distinct_event_count(events) == 2
