from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_improvements import add_candidate_support_quality_columns


def test_diagnostic_only_noncomparable_supports_are_unknown_quality() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "m1",
                "status": "success",
                "diagnostic_pyrecest_evidence_support": "particle_approximation",
            },
            {
                "model": "m2",
                "status": "success",
                "diagnostic_state_space_sparse_momentum_evidence_support": "degenerate_single_bin",
            },
            {
                "model": "m3",
                "status": "success",
                "diagnostic_state_space_trajectory_imm_evidence_support": "truncated_full_grid",
            },
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled["candidate_support_quality"].tolist() == [
        "conservative_unknown",
        "conservative_unknown",
        "conservative_unknown",
    ]
    assert not labelled["candidate_support_quality_good"].any()


def test_degenerate_canonical_support_is_unknown_quality() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "m",
                "status": "success",
                "evidence_support": "degenerate_single_bin",
            }
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled.loc[0, "candidate_support_quality"] == "conservative_unknown"
    assert not bool(labelled.loc[0, "candidate_support_quality_good"])
