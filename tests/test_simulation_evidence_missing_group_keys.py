from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import simulation_add_evidence_columns


def test_simulation_evidence_reporting_keeps_missing_group_keys() -> None:
    scores = pd.DataFrame(
        {
            "status": ["success", "success", "success", "success"],
            "session": ["Rat1/Open1", "Rat1/Open1", None, None],
            "event_index": [0, 0, 1, 1],
            "model": ["winner", "runner-up", "winner", "runner-up"],
            "log_evidence": [2.0, 1.0, 4.0, 3.0],
            "expected_model": ["winner"] * 4,
        }
    )

    scored = simulation_add_evidence_columns(scores)

    assert len(scored) == len(scores)
    missing_session = scored[scored["session"].isna()].sort_values("model")
    assert missing_session["model"].tolist() == ["runner-up", "winner"]
    assert missing_session["best_model"].unique().tolist() == ["winner"]
    assert missing_session["is_best_model"].tolist() == [False, True]
    np.testing.assert_allclose(
        missing_session["relative_log_evidence"],
        [-1.0, 0.0],
    )
