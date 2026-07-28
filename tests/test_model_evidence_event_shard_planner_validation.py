from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plan_model_evidence_event_shards import _validate_max_events, plan_event_shards  # noqa: E402


def test_plan_event_shards_rejects_negative_max_events_before_io(tmp_path):
    with pytest.raises(ValueError, match="--max-events must be non-negative"):
        plan_event_shards(
            tmp_path,
            "Rat1/Open1",
            "run",
            max_events=-1,
            event_shard_count=1,
        )


def test_validate_max_events_preserves_valid_limits():
    assert _validate_max_events(None) is None
    assert _validate_max_events(0) == 0
    assert _validate_max_events(3) == 3


def test_validate_max_events_rejects_boolean_flags():
    with pytest.raises(ValueError, match="--max-events must be non-negative"):
        _validate_max_events(True)
