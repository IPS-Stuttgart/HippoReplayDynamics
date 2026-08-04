from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_shuffle_p_values_preserve_adjacent_large_integer_event_ids() -> None:
    first_event = 2**53
    second_event = first_event + 1
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": pd.Series([first_event, second_event], dtype=object),
            "model": ["imm", "imm"],
            "log_evidence": [10.0, 20.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": pd.Series([first_event, second_event], dtype=object),
            "model": ["imm", "imm"],
            "log_evidence": [9.0, 30.0],
        }
    )

    annotated = add_shuffle_p_values(real_scores, control_scores)

    assert annotated["shuffle_count"].tolist() == [1, 1]
    assert annotated["shuffle_log_evidence_median"].tolist() == pytest.approx([9.0, 30.0])
    assert annotated["shuffle_p_value"].tolist() == pytest.approx([0.5, 1.0])
