from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm import evidence_reporting
from hipporeplayimm.evidence_reliability import add_event_reliability_flags


@pytest.mark.parametrize(
    "value",
    [b"exact_full_grid", np.bytes_("exact_full_grid")],
)
def test_byte_exact_support_is_classified_as_exact(value: object) -> None:
    comparison = evidence_reporting.evidence_comparison_from_support(value)

    assert comparison == evidence_reporting.EVIDENCE_COMPARISON_EXACT


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"truncated_full_grid", evidence_reporting.TRUNCATED_EVIDENCE_SUPPORT),
        (
            np.bytes_("degenerate_single_bin"),
            evidence_reporting.DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
        ),
        (
            b"particle_approximation",
            evidence_reporting.PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
        ),
    ],
)
def test_byte_diagnostic_support_preserves_provenance(
    value: object,
    expected: str,
) -> None:
    row = pd.Series({"diagnostic_candidate_evidence_support": value})

    assert evidence_reporting.evidence_support_from_row(row) == expected


def test_byte_degenerate_support_marks_event_unreliable() -> None:
    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "n_spikes": 4,
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
                "diagnostic_candidate_evidence_support": b"degenerate_single_bin",
            }
        ]
    )

    flagged = add_event_reliability_flags(scores)

    assert not bool(flagged.loc[0, "event_reliable"])
    assert flagged.loc[0, "event_reliability_reasons"] == "degenerate_single_bin"
