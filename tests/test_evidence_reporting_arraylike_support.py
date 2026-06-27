import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import (
    EVIDENCE_COMPARISON_LOWER_BOUND,
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
    evidence_comparison_from_support,
    evidence_support_from_row,
)


def test_evidence_support_from_row_accepts_array_like_diagnostic_cells() -> None:
    row = pd.Series(
        {
            "status": "success",
            "diagnostic_state_space_momentum_evidence_support": np.array(
                [EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT],
                dtype=object,
            ),
        }
    )

    assert evidence_support_from_row(row) == TRUNCATED_EVIDENCE_SUPPORT


def test_evidence_comparison_accepts_array_like_support_cells() -> None:
    support = np.array([EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT], dtype=object)

    assert evidence_comparison_from_support(support) == EVIDENCE_COMPARISON_LOWER_BOUND


def test_ensure_evidence_support_columns_fills_array_like_missing_support() -> None:
    frame = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "model": ["m1"],
            "log_evidence": [1.0],
            "evidence_support": [np.array([""], dtype=object)],
            "diagnostic_state_space_momentum_evidence_support": [TRUNCATED_EVIDENCE_SUPPORT],
        }
    )

    out = ensure_evidence_support_columns(frame)

    assert out.loc[0, "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert out.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_LOWER_BOUND
    assert not bool(out.loc[0, "evidence_comparable"])
