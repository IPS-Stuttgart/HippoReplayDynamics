from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import plan_model_evidence_event_shards as planner  # noqa: E402


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (0, 0),
        (4, 4),
    ],
)
def test_validate_max_events_accepts_non_negative_integers(value, expected):
    assert planner._validate_max_events(value) == expected


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "3"])
def test_validate_max_events_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        planner._validate_max_events(value)


def test_plan_event_shards_rejects_invalid_max_events_before_dataset_access(monkeypatch, tmp_path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("dataset access should not occur for an invalid max-events value")

    monkeypatch.setattr(planner, "_session_path", fail_if_called)

    with pytest.raises(ValueError, match="non-negative integer"):
        planner.plan_event_shards(
            tmp_path,
            "Rat1/Open1",
            "run",
            max_events=-1,
            event_shard_count=1,
        )
