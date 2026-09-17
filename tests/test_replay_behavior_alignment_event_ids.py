import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_behavior_alignment import (  # noqa: E402
    build_event_evidence_features,
    build_replay_behavior_alignment,
)


def _evidence_row(event_index: object, *, endpoint_x: float = 1.0) -> dict[str, object]:
    return {
        "status": "success",
        "session": "Rat1/Open1",
        "event_index": event_index,
        "model": "test-model",
        "log_evidence": 0.0,
        "diagnostic_decoded_endpoint_x": endpoint_x,
        "diagnostic_decoded_endpoint_y": 0.0,
        "evidence_comparable": True,
    }


def _features(evidence: pd.DataFrame) -> pd.DataFrame:
    return build_event_evidence_features(
        evidence,
        required_models=("test-model",),
        trajectory_models=("test-model",),
    )


def test_behavior_alignment_keeps_adjacent_decimal_form_ids_above_float_exact_range_distinct():
    first = 2**53
    second = first + 1
    evidence = pd.DataFrame(
        [
            _evidence_row(f"{first}.0", endpoint_x=1.0),
            _evidence_row(f"{second}.0", endpoint_x=2.0),
        ]
    )

    features = _features(evidence)

    assert features["event_index"].tolist() == [first, second]
    assert features["decoded_endpoint_x"].tolist() == [1.0, 2.0]


def test_behavior_alignment_rejects_fractional_event_ids_instead_of_truncating_them():
    evidence = pd.DataFrame([_evidence_row("3.5")])

    with pytest.raises(ValueError, match="event_index.*integer"):
        _features(evidence)


def test_behavior_alignment_normalizes_context_event_ids_exactly_before_join():
    event_index = 2**53 + 1
    features = _features(pd.DataFrame([_evidence_row(f"{event_index}.0")]))
    context = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": f"{event_index}.0",
                "current_x": 0.0,
                "current_y": 0.0,
                "previous_x": -1.0,
                "previous_y": 0.0,
                "future_x": 2.0,
                "future_y": 0.0,
                "active_goal_x": 3.0,
                "active_goal_y": 0.0,
                "nearest_current_well_x": 3.0,
                "nearest_current_well_y": 0.0,
            }
        ]
    )

    alignment = build_replay_behavior_alignment(features, context)

    assert alignment["event_index"].tolist() == [event_index]
    assert alignment["current_x"].tolist() == [0.0]
    assert alignment["alignment_with_next_movement"].tolist() == [1.0]
