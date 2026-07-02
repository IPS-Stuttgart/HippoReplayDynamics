from __future__ import annotations

import pandas as pd

from hipporeplayimm.evidence_reporting import ensure_evidence_support_columns


def test_failed_status_overrides_stale_exact_support_label() -> None:
    rows = pd.DataFrame(
        [
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "model": "failed-stale-exact",
                "status": "failure",
                "log_evidence": 100.0,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": True,
            },
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "model": "success-exact",
                "status": "success",
                "log_evidence": 0.0,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": True,
            },
        ]
    )

    supported = ensure_evidence_support_columns(rows)
    failed = supported.loc[supported["model"] == "failed-stale-exact"].iloc[0]
    successful = supported.loc[supported["model"] == "success-exact"].iloc[0]

    assert failed["evidence_support"] == "not_scored"
    assert failed["evidence_comparison"] == "not_scored"
    assert not bool(failed["evidence_comparable"])
    assert successful["evidence_support"] == "exact_full_grid"
    assert bool(successful["evidence_comparable"])
