from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_quality_gates import quality_gate_summary


def _object_scalar(value: object) -> np.ndarray:
    scalar = np.empty((), dtype=object)
    scalar[()] = value
    return scalar


@pytest.mark.parametrize(
    "minimum",
    [
        True,
        np.bool_(False),
        np.asarray(True, dtype=object),
        _object_scalar(np.asarray(True)),
        0,
        -1,
        1.5,
        "1.5",
        "2.0",
        "2e0",
        b"2.0",
        np.nan,
        np.inf,
        np.asarray([2]),
        np.complex128(2.0 + 0.0j),
    ],
)
def test_quality_gate_summary_rejects_invalid_exact_model_minimum_before_empty_return(
    minimum: object,
) -> None:
    with pytest.raises(ValueError, match="min_exact_models_per_event"):
        quality_gate_summary(
            pd.DataFrame(),
            min_exact_models_per_event=minimum,
        )


@pytest.mark.parametrize(
    "minimum",
    [
        1,
        2.0,
        "2",
        b"2",
        np.int64(2),
        np.asarray(2),
        np.asarray(2, dtype=object),
        Decimal("2"),
    ],
)
def test_quality_gate_summary_accepts_positive_integer_like_minimum(
    minimum: object,
) -> None:
    summary = quality_gate_summary(
        pd.DataFrame(),
        min_exact_models_per_event=minimum,
    )

    assert summary.to_dict("records") == [
        {
            "gate": "nonempty_scores",
            "status": "fail",
            "value": 0,
            "note": "No score rows.",
        }
    ]


@pytest.mark.parametrize(
    "minimum",
    [
        True,
        np.bool_(False),
        np.asarray(True, dtype=object),
        _object_scalar(np.asarray(True)),
        -0.01,
        1.01,
        np.nan,
        np.inf,
        -np.inf,
        np.asarray([0.95]),
        np.complex128(0.95 + 0.0j),
        "0.95",
    ],
)
def test_quality_gate_summary_rejects_invalid_candidate_fraction_before_empty_return(
    minimum: object,
) -> None:
    with pytest.raises(ValueError, match="min_candidate_good_fraction"):
        quality_gate_summary(
            pd.DataFrame(),
            min_candidate_good_fraction=minimum,
        )


@pytest.mark.parametrize(
    "minimum",
    [
        0,
        1,
        0.0,
        1.0,
        np.float64(0.95),
        np.asarray(0.95),
        np.asarray(0.95, dtype=object),
        Decimal("0.95"),
    ],
)
def test_quality_gate_summary_accepts_unit_interval_candidate_fraction(
    minimum: object,
) -> None:
    summary = quality_gate_summary(
        pd.DataFrame(),
        min_candidate_good_fraction=minimum,
    )

    assert summary.loc[0, "gate"] == "nonempty_scores"
    assert summary.loc[0, "status"] == "fail"


def test_quality_gate_summary_uses_canonical_validated_thresholds() -> None:
    scores = pd.DataFrame(
        {
            "model": ["diffusion", "stationary"],
            "log_evidence": [4.0, 1.5],
            "status": ["success", "success"],
            "diagnostic_candidate_evidence_support": [
                "exact_full_grid",
                "exact_full_grid",
            ],
        }
    )

    summary = quality_gate_summary(
        scores,
        min_exact_models_per_event="2",
        min_candidate_good_fraction=np.asarray(0.95),
    )

    exact_gate = summary.loc[
        summary["gate"].eq("events_with_min_exact_models")
    ].iloc[0]
    assert exact_gate["status"] == "pass"
    assert float(exact_gate["value"]) == pytest.approx(1.0)
