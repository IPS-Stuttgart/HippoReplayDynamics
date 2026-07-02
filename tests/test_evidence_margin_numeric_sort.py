from __future__ import annotations

import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import evidence_margin_table


def test_evidence_margin_table_sorts_string_evidence_numerically() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0],
            "model": ["diffusion", "momentum", "stationary"],
            "log_evidence": ["9.0", "10.0", "not-a-number"],
            "status": ["success", "success", "success"],
            "evidence_comparable": [True, True, True],
        }
    )

    margins = evidence_margin_table(scores)

    assert len(margins) == 1
    assert margins.loc[0, "best_model_by_evidence"] == "momentum"
    assert margins.loc[0, "second_best_model_by_evidence"] == "diffusion"
    assert margins.loc[0, "best_log_evidence"] == 10.0
    assert margins.loc[0, "second_best_log_evidence"] == 9.0
    assert margins.loc[0, "evidence_margin_to_second_best"] == 1.0
    assert margins.loc[0, "models_compared"] == 2
