from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_replay_dynamics_axis import (
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
    _read_csv_preserving_event_index,
    build_event_dynamics_axis,
)


MODELS = (STATIONARY, DIFFUSION, FRAGMENTED, FIRST_ORDER_IMM, MOMENTUM_EXACT)


def _event_scores(event_index: object) -> list[dict[str, object]]:
    return [
        {
            "status": "success",
            "evidence_comparable": True,
            "session": "Rat1/Open1",
            "event_index": event_index,
            "model": model,
            "log_evidence": float(index),
        }
        for index, model in enumerate(MODELS)
    ]


def test_dynamics_axis_preserves_adjacent_decimal_event_ids_above_2_to_53(
    tmp_path: Path,
) -> None:
    lower = 2**53
    upper = lower + 1
    evidence = pd.DataFrame(
        [
            *_event_scores(f"{lower}.0"),
            *_event_scores(f"{upper}.0"),
        ]
    )
    path = tmp_path / "event_model_evidence.csv"
    evidence.to_csv(path, index=False)

    loaded = _read_csv_preserving_event_index(path)
    event_axis = build_event_dynamics_axis(loaded, covariates=())

    assert loaded["event_index"].dtype.name == "string"
    assert event_axis["event_index"].tolist() == [lower, upper]


def test_dynamics_axis_rejects_fractional_numeric_event_ids() -> None:
    evidence = pd.DataFrame(_event_scores(1.5))

    with pytest.raises(ValueError, match="integer-valued"):
        build_event_dynamics_axis(evidence, covariates=())


def test_dynamics_axis_rejects_unsafe_preparsed_large_float_event_ids() -> None:
    evidence = pd.DataFrame(_event_scores(float(2**53)))

    with pytest.raises(ValueError, match="at or above 2\\*\\*53 is unsafe"):
        build_event_dynamics_axis(evidence, covariates=())


def test_dynamics_axis_rejects_boolean_event_ids() -> None:
    evidence = pd.DataFrame(_event_scores(True))

    with pytest.raises(ValueError, match="not booleans"):
        build_event_dynamics_axis(evidence, covariates=())
