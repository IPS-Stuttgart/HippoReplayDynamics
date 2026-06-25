from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.recovery_diagnostics import _successful_finite_scores


def test_recovery_diagnostics_treats_legacy_missing_status_as_success() -> None:
    scores = pd.DataFrame(
        {
            "status": [np.nan, "", "Success", "failed", "success"],
            "log_evidence": [1.0, 2.0, 3.0, 4.0, float("nan")],
        }
    )

    successful = _successful_finite_scores(scores)

    assert successful["log_evidence"].tolist() == [1.0, 2.0, 3.0]
