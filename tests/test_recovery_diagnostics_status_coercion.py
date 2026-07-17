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


def test_recovery_diagnostics_ignores_unrepresentable_log_evidence() -> None:
    huge = 10**400
    scores = pd.DataFrame(
        {
            "status": ["success", np.nan, "success", "success"],
            "model": ["valid", "positive-overflow", "negative-overflow", "infinite"],
        }
    )
    scores["log_evidence"] = pd.Series(
        [1.5, huge, -huge, float("inf")],
        dtype=object,
    )

    successful = _successful_finite_scores(scores)

    assert successful["model"].tolist() == ["valid"]
    assert successful["log_evidence"].tolist() == [1.5]
