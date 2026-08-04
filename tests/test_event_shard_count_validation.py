from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import plan_kd_event_shards as kd_planner  # noqa: E402
import plan_model_evidence_event_shards as model_planner  # noqa: E402


@pytest.mark.parametrize("planner", [model_planner, kd_planner])
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 1.5, "2", np.array([2], dtype=int)],
)
def test_event_shard_planners_reject_noninteger_shard_counts(planner, value: object) -> None:
    with pytest.raises(TypeError, match="--event-shard-count.*integer scalar"):
        planner._event_chunks([0, 1], value)


@pytest.mark.parametrize("planner", [model_planner, kd_planner])
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 1.5, "2", np.array([2], dtype=int)],
)
def test_event_shard_planners_reject_noninteger_max_events(planner, value: object) -> None:
    with pytest.raises(TypeError, match="--max-events.*integer scalar"):
        planner._validate_max_events(value)


def _stub_session_loading(monkeypatch: pytest.MonkeyPatch, planner) -> None:
    monkeypatch.setattr(planner, "_session_path", lambda root, session_id: root / session_id)
    monkeypatch.setattr(planner, "_check_session", lambda session_dir: None)
    monkeypatch.setattr(
        planner,
        "load_replay_session",
        lambda session_dir: SimpleNamespace(session_id="Rat1/Open1"),
    )


def test_model_evidence_plan_canonicalizes_numpy_integer_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_session_loading(monkeypatch, model_planner)
    monkeypatch.setattr(model_planner, "_events", lambda events, session: list(range(5)))

    plan = model_planner.plan_event_shards(
        tmp_path,
        "Rat1/Open1",
        "run",
        max_events=np.int64(3),
        event_shard_count=np.int64(2),
    )

    assert plan["max_events"] == 3
    assert type(plan["max_events"]) is int
    assert plan["requested_event_shard_count"] == 2
    assert type(plan["requested_event_shard_count"]) is int
    assert json.loads(json.dumps(plan))["max_events"] == 3


def test_kd_plan_canonicalizes_all_numpy_integer_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_session_loading(monkeypatch, kd_planner)
    monkeypatch.setattr(
        kd_planner,
        "_events",
        lambda events, session, max_events: list(range(5))[:max_events],
    )

    plan = kd_planner.plan_event_shards(
        tmp_path,
        "Rat1/Open1",
        "run",
        max_events=np.int64(3),
        event_shard_count=np.int64(2),
        momentum_shard_count=np.int64(2),
    )

    for key, expected in (
        ("max_events", 3),
        ("requested_event_shard_count", 2),
        ("momentum_shard_count", 2),
    ):
        assert plan[key] == expected
        assert type(plan[key]) is int
    assert json.loads(json.dumps(plan))["grid_shard_indices"] == [0, 1]
