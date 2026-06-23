import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import evidence_margin_table


def test_advanced_diagnostics_keep_missing_legacy_status_rows():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0],
            "model": ["stationary", "diffusion", "failed-debug"],
            "log_evidence": [1.0, 4.0, 100.0],
            "status": [np.nan, "", "failed"],
            "evidence_comparable": [True, True, True],
        }
    )

    margins = evidence_margin_table(scores)

    assert len(margins) == 1
    assert margins.loc[0, "best_model_by_evidence"] == "diffusion"
    assert margins.loc[0, "second_best_model_by_evidence"] == "stationary"
    assert margins.loc[0, "models_compared"] == 2
    assert margins.loc[0, "best_log_evidence"] == 4.0
