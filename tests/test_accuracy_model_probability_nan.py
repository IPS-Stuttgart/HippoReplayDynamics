from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.accuracy_upgrades import model_probability_diagnostics


def test_model_probability_diagnostics_ignores_nan_evidence_for_best_model() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0],
            "model": ["nan-evidence", "low", "high"],
            "log_evidence": [np.nan, 1.0, 3.0],
            "status": ["success", "success", "success"],
            "evidence_comparable": [True, True, True],
        }
    )

    diagnostics = model_probability_diagnostics(scores)

    assert diagnostics.shape[0] == 1
    assert diagnostics.loc[0, "models"] == 2
    assert diagnostics.loc[0, "best_model"] == "high"
    assert diagnostics.loc[0, "best_log_evidence"] == 3.0
