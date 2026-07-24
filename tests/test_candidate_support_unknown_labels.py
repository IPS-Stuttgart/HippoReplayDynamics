from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_improvements import add_candidate_support_quality_columns


def test_unrecognized_candidate_support_is_not_labelled_exact() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "approximate-model",
                "status": "success",
                "evidence_support": "laplace_approximation",
            },
            {
                "model": "mixed-support-model",
                "status": "success",
                "evidence_support": "exact_full_grid",
                "diagnostic_custom_evidence_support": "experimental_subset",
            },
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled["candidate_support_quality"].tolist() == [
        "conservative_unknown",
        "conservative_unknown",
    ]
    assert labelled["candidate_support_quality_good"].tolist() == [False, False]


def test_missing_legacy_support_and_explicit_exact_support_remain_exact() -> None:
    rows = pd.DataFrame(
        [
            {"model": "legacy", "status": "success"},
            {
                "model": "exact",
                "status": "success",
                "evidence_support": "exact_full_grid",
            },
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled["candidate_support_quality"].tolist() == [
        "exact_or_not_pruned",
        "exact_or_not_pruned",
    ]
    assert labelled["candidate_support_quality_good"].tolist() == [True, True]
