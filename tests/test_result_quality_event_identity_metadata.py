from __future__ import annotations

import pandas as pd

from hipporeplayimm import result_quality_audit as audit


def test_result_quality_normalizes_event_identity_metadata() -> None:
    scores = pd.DataFrame(
        {
            "session": [bytearray(b"Rat1/Open1"), memoryview(b"Rat1/Open1")],
            "event_id": [[7], (7,)],
            "model": ["stationary", "diffusion"],
            "log_evidence": [2.0, 1.0],
        }
    )

    normalized = audit._score_table_with_log_evidence_alias(scores)

    assert normalized["session"].tolist() == ["Rat1/Open1", "Rat1/Open1"]
    assert normalized["event_id"].tolist() == [("sequence", (7,))] * 2
    assert audit.event_group_columns(normalized) == ["session", "event_id"]
    assert audit._event_count(scores) == 1
