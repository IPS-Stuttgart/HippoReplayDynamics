from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.report_replay_commitment_composition_hypothesis import (
    circularly_shift_predictor_within_session,
    classify_hypothesis,
)


def test_circular_predictor_shift_preserves_values_within_each_session() -> None:
    frame = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4 + ["Rat2/Open1"] * 4,
            "event_index": list(range(4)) * 2,
            "delta_momentum_minus_imm": np.arange(8, dtype=float),
        }
    )

    shifted = circularly_shift_predictor_within_session(
        frame,
        rng=np.random.default_rng(2),
    )

    for session, group in frame.groupby("session"):
        shifted_group = shifted[shifted["session"].eq(session)]
        assert sorted(group["delta_momentum_minus_imm"]) == sorted(
            shifted_group["delta_momentum_minus_imm"]
        )
        assert not np.array_equal(
            group["delta_momentum_minus_imm"],
            shifted_group["delta_momentum_minus_imm"],
        )


def test_hypothesis_classification_keeps_partial_outcomes_distinct() -> None:
    assert (
        classify_hypothesis(
            composition_supported=False,
            commitment_supported=False,
            boundary_supported=False,
            external_supported=False,
        )
        == "neither_primary_specialization_supported"
    )
    assert (
        classify_hypothesis(
            composition_supported=True,
            commitment_supported=False,
            boundary_supported=True,
            external_supported=False,
        )
        == "composition_only"
    )
    assert (
        classify_hypothesis(
            composition_supported=True,
            commitment_supported=True,
            boundary_supported=True,
            external_supported=True,
        )
        == "strong_support"
    )
