from __future__ import annotations

import pandas as pd

import hipporeplayimm
import hipporeplayimm.evidence_reporting as evidence_reporting


def test_large_nonzero_integer_evidence_flag_does_not_overflow() -> None:
    hipporeplayimm.apply_runtime_patches()
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "exact-row",
                "log_evidence": 1.0,
                "evidence_comparable": 10**400,
            },
            {
                "status": "success",
                "model": "legacy-noncomparable-row",
                "log_evidence": 1.0,
                "evidence_comparable": 0,
            },
        ]
    )

    scored = evidence_reporting.ensure_evidence_support_columns(rows)

    exact = scored[scored["model"] == "exact-row"].iloc[0]
    legacy_noncomparable = scored[
        scored["model"] == "legacy-noncomparable-row"
    ].iloc[0]
    assert exact["evidence_support"] == evidence_reporting.EXACT_EVIDENCE_SUPPORT
    assert bool(exact["evidence_comparable"])
    assert (
        legacy_noncomparable["evidence_support"]
        == evidence_reporting.EVIDENCE_COMPARISON_UNKNOWN
    )
    assert not bool(legacy_noncomparable["evidence_comparable"])
