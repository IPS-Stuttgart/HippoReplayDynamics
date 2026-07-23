from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvements import add_candidate_support_quality_columns


def test_byte_encoded_noncomparable_support_is_not_mislabelled_exact() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "particle",
                "status": "success",
                "evidence_support": b"particle_approximation",
            },
            {
                "model": "custom",
                "status": "success",
                "diagnostic_custom_evidence_support": np.array(
                    [np.bytes_("unknown_noncomparable")],
                    dtype=object,
                ),
            },
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled["candidate_support_quality"].tolist() == [
        "conservative_unknown",
        "conservative_unknown",
    ]
    assert not labelled["candidate_support_quality_good"].any()


def test_byte_encoded_success_and_exact_labels_remain_exact() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "exact",
                "status": np.bytes_("success"),
                "evidence_support": np.bytes_("exact_full_grid"),
            }
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled.loc[0, "candidate_support_quality"] == "exact_or_not_pruned"
    assert bool(labelled.loc[0, "candidate_support_quality_good"])


def test_memoryview_success_and_exact_labels_remain_exact() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "exact",
                "status": memoryview(b"success"),
                "evidence_support": memoryview(b"exact_full_grid"),
            }
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled.loc[0, "candidate_support_quality"] == "exact_or_not_pruned"
    assert bool(labelled.loc[0, "candidate_support_quality_good"])


def test_bytearray_noncomparable_support_remains_unknown() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "particle",
                "status": bytearray(b"success"),
                "evidence_support": bytearray(b"particle_approximation"),
            }
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled.loc[0, "candidate_support_quality"] == "conservative_unknown"
    assert not bool(labelled.loc[0, "candidate_support_quality_good"])
