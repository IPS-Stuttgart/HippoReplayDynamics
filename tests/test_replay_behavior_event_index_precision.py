import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_behavior_alignment import (  # noqa: E402
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
    REQUIRED_EXACT_CORE_MODELS,
    build_event_evidence_features,
    build_replay_behavior_alignment,
)


def _evidence_rows(event_index: object, endpoint_x: float) -> list[dict[str, object]]:
    log_evidence = {
        REQUIRED_EXACT_CORE_MODELS[0]: 0.0,
        REQUIRED_EXACT_CORE_MODELS[1]: 1.0,
        REQUIRED_EXACT_CORE_MODELS[2]: 2.0,
        FIRST_ORDER_IMM: 4.0,
        MOMENTUM_EXACT: 3.0,
    }
    return [
        {
            "status": "success",
            "session": "Rat1/Open1",
            "event_index": event_index,
            "model": model,
            "log_evidence": value,
            "evidence_comparable": True,
            "diagnostic_decoded_endpoint_x": endpoint_x,
            "diagnostic_decoded_endpoint_y": 0.0,
        }
        for model, value in log_evidence.items()
    ]


def _context(event_index: object, current_x: float) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": event_index,
        "current_x": current_x,
        "current_y": 0.0,
        "previous_x": current_x - 1.0,
        "previous_y": 0.0,
        "future_x": current_x + 1.0,
        "future_y": 0.0,
        "active_goal_x": current_x + 2.0,
        "active_goal_y": 0.0,
        "nearest_current_well_x": current_x + 2.0,
        "nearest_current_well_y": 0.0,
    }


def test_behavior_alignment_preserves_adjacent_decimal_event_ids_above_2_to_53() -> None:
    lower = 2**53
    upper = lower + 1
    evidence = pd.DataFrame(
        [
            *_evidence_rows(f"{lower}.0", 10.0),
            *_evidence_rows(f"{upper}.0", 20.0),
        ]
    )
    context = pd.DataFrame(
        [
            _context(f"{lower}.0", 100.0),
            _context(f"{upper}.0", 200.0),
        ]
    )

    features = build_event_evidence_features(evidence)
    alignment = build_replay_behavior_alignment(features, context).sort_values("event_index")

    assert features.sort_values("event_index")["event_index"].tolist() == [lower, upper]
    assert alignment["event_index"].tolist() == [lower, upper]
    assert alignment["current_x"].tolist() == [100.0, 200.0]


def test_behavior_alignment_rejects_preparsed_large_float_event_ids() -> None:
    evidence = pd.DataFrame(_evidence_rows(float(2**53), 10.0))

    with pytest.raises(ValueError, match=r"floating-point event_index at or above 2\*\*53 is unsafe"):
        build_event_evidence_features(evidence)
