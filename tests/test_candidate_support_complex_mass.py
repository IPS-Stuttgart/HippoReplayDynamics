from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import hipporeplayimm
from hipporeplayimm.result_improvements import (
    CANDIDATE_SUPPORT_UNKNOWN,
    add_candidate_support_quality_columns,
    candidate_support_quality,
)


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def test_complex_candidate_mass_is_not_cast_into_good_support_quality() -> None:
    hipporeplayimm.apply_runtime_patches()
    rows = pd.DataFrame(
        [
            {
                "model": "direct-complex",
                "status": "success",
                "evidence_support": "truncated_full_grid",
                "min_candidate_log_mass": np.complex128(-0.005 + 0.25j),
            },
            {
                "model": "nested-complex",
                "status": "success",
                "evidence_support": "truncated_full_grid",
                "min_candidate_log_mass": _nested_object_scalar(
                    np.complex128(-0.005 + 0.25j)
                ),
            },
            {
                "model": "real",
                "status": "success",
                "evidence_support": "truncated_full_grid",
                "min_candidate_log_mass": -0.005,
            },
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        labelled = add_candidate_support_quality_columns(rows)

    assert np.isnan(labelled.loc[0, "candidate_min_log_mass"])
    assert np.isnan(labelled.loc[1, "candidate_min_log_mass"])
    assert labelled["candidate_support_quality"].tolist() == [
        CANDIDATE_SUPPORT_UNKNOWN,
        CANDIDATE_SUPPORT_UNKNOWN,
        "conservative_good",
    ]
    assert labelled["candidate_support_quality_good"].tolist() == [
        False,
        False,
        True,
    ]


def test_direct_complex_candidate_mass_argument_is_unknown_without_warning() -> None:
    hipporeplayimm.apply_runtime_patches()
    row = pd.Series(
        {
            "status": "success",
            "evidence_support": "truncated_full_grid",
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quality = candidate_support_quality(
            row,
            min_log_mass=np.complex128(-0.005 + 0.25j),
        )

    assert quality == CANDIDATE_SUPPORT_UNKNOWN
