from __future__ import annotations

import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import evidence_reporting
from hipporeplayimm.evidence_reliability import add_event_reliability_flags


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (bytearray(b"exact_full_grid"), evidence_reporting.EVIDENCE_COMPARISON_EXACT),
        (
            bytearray(b"truncated_full_grid"),
            evidence_reporting.EVIDENCE_COMPARISON_LOWER_BOUND,
        ),
        (
            bytearray(b"particle_approximation"),
            evidence_reporting.EVIDENCE_COMPARISON_PARTICLE_APPROXIMATION,
        ),
    ],
)
def test_bytearray_support_scalars_preserve_comparison_scope(
    value: bytearray,
    expected: str,
) -> None:
    hipporeplayimm.apply_runtime_patches()

    assert evidence_reporting.evidence_comparison_from_support(value) == expected


def test_bytearray_diagnostic_support_preserves_provenance() -> None:
    hipporeplayimm.apply_runtime_patches()
    row = pd.Series(
        {"diagnostic_candidate_evidence_support": bytearray(b"degenerate_single_bin")}
    )

    assert (
        evidence_reporting.evidence_support_from_row(row)
        == evidence_reporting.DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT
    )


def test_bytearray_degenerate_support_marks_event_unreliable() -> None:
    hipporeplayimm.apply_runtime_patches()
    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "n_spikes": 4,
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
                "diagnostic_candidate_evidence_support": bytearray(
                    b"degenerate_single_bin"
                ),
            }
        ]
    )

    flagged = add_event_reliability_flags(scores)

    assert not bool(flagged.loc[0, "event_reliable"])
    assert flagged.loc[0, "event_reliability_reasons"] == "degenerate_single_bin"
