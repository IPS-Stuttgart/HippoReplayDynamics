from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_improvements import add_candidate_support_quality_columns


def test_non_success_and_non_comparable_rows_are_unknown_support_quality() -> None:
    rows = pd.DataFrame(
        [
            {"model": "m1", "status": "error", "evidence_support": "not_scored"},
            {"model": "m2", "status": "success", "evidence_support": "particle_approximation"},
            {"model": "m3", "status": "success", "evidence_support": "unknown_noncomparable"},
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled["candidate_support_quality"].tolist() == [
        "conservative_unknown",
        "conservative_unknown",
        "conservative_unknown",
    ]
    assert not labelled["candidate_support_quality_good"].any()


def test_non_comparable_diagnostic_evidence_support_is_unknown_quality() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "pyrecest-goal-particle",
                "status": "success",
                "diagnostic_pyrecest_evidence_support": "particle_approximation",
            },
            {
                "model": "custom-model",
                "status": "success",
                "diagnostic_custom_evidence_support": "unknown_noncomparable",
            },
            {
                "model": "state-space-displacement-imm",
                "status": "success",
                "diagnostic_state_space_displacement_imm_evidence_support": "truncated_full_grid",
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


def test_exact_success_rows_remain_good_support_quality() -> None:
    rows = pd.DataFrame(
        [{"model": "m", "status": "success", "evidence_support": "exact_full_grid"}]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled.loc[0, "candidate_support_quality"] == "exact_or_not_pruned"
    assert bool(labelled.loc[0, "candidate_support_quality_good"])
