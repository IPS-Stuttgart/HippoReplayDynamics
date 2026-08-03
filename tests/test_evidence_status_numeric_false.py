from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.evidence_reporting import (
    EVIDENCE_COMPARISON_UNKNOWN,
    ensure_evidence_support_columns,
)
from hipporeplayimm.evidence_status_coercion import _is_explicit_false_value


@pytest.mark.parametrize("encoded_false", ["0e0", "-0", "+0.000", "0E+999"])
def test_numeric_zero_comparable_flags_remain_noncomparable(encoded_false: str) -> None:
    rows = pd.DataFrame(
        [
            {
                "status": "success",
                "model": "legacy-candidate-pruned-row",
                "log_evidence": 100.0,
                "evidence_comparable": encoded_false,
            }
        ]
    )

    scored = ensure_evidence_support_columns(rows)

    assert scored.loc[0, "evidence_support"] == EVIDENCE_COMPARISON_UNKNOWN
    assert not bool(scored.loc[0, "evidence_comparable"])
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_UNKNOWN


def test_numeric_false_detection_does_not_underflow_tiny_nonzero_strings() -> None:
    assert not _is_explicit_false_value("1e-4000")
