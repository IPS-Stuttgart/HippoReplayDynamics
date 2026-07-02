from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_quality_audit_scope_patch import _scoped_event_group_columns


def test_textual_missing_optional_metadata_does_not_scope_result_quality_events() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["diffusion", "momentum"],
            "window_index": ["nan", " <NA> "],
            "benchmark_random_seed": ["", "None"],
        }
    )

    assert _scoped_event_group_columns(scores) == ["session", "event_index"]


def test_observed_optional_metadata_still_scopes_result_quality_events() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["diffusion", "momentum"],
            "window_index": ["nan", "0"],
        }
    )

    assert _scoped_event_group_columns(scores) == ["session", "event_index", "window_index"]
