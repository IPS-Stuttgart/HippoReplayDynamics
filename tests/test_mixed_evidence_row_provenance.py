import numpy as np
import pandas as pd

from hipporeplayimm import evidence_reporting as reporting


def test_missing_heldout_value_does_not_change_raw_truncated_support():
    row = pd.Series(
        {
            "diagnostic_candidate_evidence_support": reporting.TRUNCATED_EVIDENCE_SUPPORT,
            "heldout_log_likelihood": np.nan,
        }
    )

    assert (
        reporting.evidence_support_from_row(row)
        == reporting.TRUNCATED_EVIDENCE_SUPPORT
    )


def test_mixed_raw_and_heldout_rows_keep_row_specific_support():
    frame = pd.DataFrame(
        {
            "log_evidence": [-10.0, -11.0],
            "evidence_support": [
                reporting.TRUNCATED_EVIDENCE_SUPPORT,
                reporting.TRUNCATED_EVIDENCE_SUPPORT,
            ],
            "heldout_log_likelihood": [np.nan, -3.0],
        },
        index=["raw", "heldout"],
    )

    result = reporting.ensure_evidence_support_columns(frame)

    assert (
        result.loc["raw", "evidence_support"]
        == reporting.TRUNCATED_EVIDENCE_SUPPORT
    )
    assert (
        result.loc["raw", "evidence_comparison"]
        == reporting.EVIDENCE_COMPARISON_LOWER_BOUND
    )
    assert (
        result.loc["heldout", "evidence_support"]
        == reporting.RESTRICTED_HELDOUT_EVIDENCE_SUPPORT
    )
    assert (
        result.loc["heldout", "evidence_comparison"]
        == reporting.EVIDENCE_COMPARISON_RESTRICTED_DIFFERENCE
    )
