from io import StringIO

import pandas as pd
import pytest

from hipporeplayimm.evidence_reporting import (
    EVIDENCE_COMPARISON_DEGENERATE,
    EVIDENCE_COMPARISON_EXACT,
    EVIDENCE_COMPARISON_LOWER_BOUND,
    EVIDENCE_COMPARISON_UNKNOWN,
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
)


def _score_row(support: object) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "status": "success",
                "model": "candidate-model",
                "log_evidence": -1.0,
                "diagnostic_candidate_evidence_support": support,
            }
        ]
    )


@pytest.mark.parametrize(
    "support",
    [
        "exact_full_grid,truncated_full_grid",
        "['exact_full_grid', 'truncated_full_grid']",
        '["exact_full_grid"; "truncated_full_grid"]',
    ],
)
def test_serialized_multi_support_preserves_truncated_classification(support: str):
    scored = ensure_evidence_support_columns(_score_row(support))

    assert scored.loc[0, "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_LOWER_BOUND
    assert not bool(scored.loc[0, "evidence_comparable"])


def test_csv_round_trip_preserves_nonexact_support_priority():
    original = _score_row([EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT])
    buffer = StringIO()
    original.to_csv(buffer, index=False)
    buffer.seek(0)

    scored = ensure_evidence_support_columns(pd.read_csv(buffer))

    assert scored.loc[0, "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_LOWER_BOUND
    assert not bool(scored.loc[0, "evidence_comparable"])


def test_serialized_single_known_support_is_recovered():
    exact = ensure_evidence_support_columns(_score_row("['exact_full_grid']"))
    degenerate = ensure_evidence_support_columns(_score_row("['degenerate_single_bin']"))

    assert exact.loc[0, "evidence_support"] == EXACT_EVIDENCE_SUPPORT
    assert exact.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_EXACT
    assert bool(exact.loc[0, "evidence_comparable"])
    assert degenerate.loc[0, "evidence_support"] == "degenerate_single_bin"
    assert degenerate.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_DEGENERATE
    assert not bool(degenerate.loc[0, "evidence_comparable"])


def test_unknown_scalar_support_remains_unknown():
    scored = ensure_evidence_support_columns(_score_row("custom_unknown_support"))

    assert scored.loc[0, "evidence_support"] == EVIDENCE_COMPARISON_UNKNOWN
    assert scored.loc[0, "evidence_comparison"] == EVIDENCE_COMPARISON_UNKNOWN
    assert not bool(scored.loc[0, "evidence_comparable"])
