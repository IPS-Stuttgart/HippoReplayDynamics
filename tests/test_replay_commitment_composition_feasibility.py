from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from scripts.audit_replay_commitment_composition_feasibility import (
    BEHAVIOR_COVERAGE_OUTPUT,
    EXACT_CORE_MODELS,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    GATE_OUTPUT,
    MANIFEST_OUTPUT,
    MOMENTUM,
    PRIMARY_AXIS_OUTPUT,
    audit_hc11_awake_suitability,
    build_behavior_coverage,
    build_frozen_event_table,
    build_gates,
    choose_primary_momentum_axis,
)


def _event_rows(
    session: str,
    event_index: int,
    scores: dict[str, float],
) -> list[dict[str, object]]:
    return [
        {
            "status": "success",
            "session": session,
            "event_index": event_index,
            "model": model,
            "log_evidence": scores[model],
            "n_time": 20,
            "n_spikes": 30,
            "evidence_comparable": True,
        }
        for model in EXACT_CORE_MODELS
    ]


def _scores(best: str, *, margin: float = 10.0) -> dict[str, float]:
    scores = {model: -20.0 - index for index, model in enumerate(EXACT_CORE_MODELS)}
    scores[best] = 0.0
    if best == FIRST_ORDER_IMM:
        scores[FRAGMENTED] = -margin
    elif best == MOMENTUM:
        scores[FIRST_ORDER_IMM] = -margin
    return scores


def test_primary_axis_falls_back_before_behavior_when_momentum_is_sparse() -> None:
    rows: list[dict[str, object]] = []
    for index in range(7):
        rows.extend(_event_rows(f"Rat{index % 3 + 1}/Open1", index, _scores(MOMENTUM)))
    for index in range(7, 12):
        rows.extend(_event_rows(f"Rat{index % 4 + 1}/Open1", index, _scores(FIRST_ORDER_IMM)))
    events = build_frozen_event_table(pd.DataFrame(rows))

    decision = choose_primary_momentum_axis(events).iloc[0]

    assert int(events["confident_momentum_win"].sum()) == 7
    assert int(events.loc[events["confident_momentum_win"], "rat"].nunique()) == 3
    assert decision["primary_predictor"] == "delta_momentum_minus_imm"
    assert decision["categorical_model_classes_role"] == "secondary_descriptive_only"
    assert bool(decision["decision_frozen_before_behavior"])
    assert set(events.loc[events["clean_imm"], "analysis_role"]) == {"clean_imm"}


def test_primary_axis_uses_categorical_class_only_after_count_gate() -> None:
    rows: list[dict[str, object]] = []
    for index in range(10):
        rows.extend(_event_rows(f"Rat{index % 3 + 1}/Open1", index, _scores(MOMENTUM)))
    events = build_frozen_event_table(pd.DataFrame(rows))

    decision = choose_primary_momentum_axis(events).iloc[0]

    assert decision["primary_predictor"] == "confident_momentum_class"
    assert bool(decision["categorical_momentum_primary_ready"])


@dataclass
class _Ripple:
    start: float
    end: float
    peak: float


class _Session:
    def __init__(self) -> None:
        time = np.arange(0.0, 100.0, 0.1)
        x = np.zeros_like(time)
        before = time < 29.0
        after = time >= 32.0
        x[before] = 2.0 * time[before]
        x[(time >= 29.0) & (time < 32.0)] = 58.0
        x[after] = 58.0 + 10.0 * (time[after] - 32.0)
        y = np.zeros_like(time)
        speed = np.gradient(x, time)
        self.position = np.column_stack([time, x, y, speed])
        self.run_times = np.array([[0.0, 100.0]])
        self.well_sequence = np.array([[20.0, 1.0], [50.0, 2.0]])

    def ripple(self, event_index: int) -> _Ripple:
        assert event_index == 5
        return _Ripple(start=29.95, end=30.05, peak=30.0)


def test_behavior_coverage_is_separate_and_detects_future_route(tmp_path: Path) -> None:
    events = pd.DataFrame([{"session": "Rat1/Open1", "rat": "Rat1", "event_index": 5}])
    coverage = build_behavior_coverage(
        events,
        tmp_path,
        session_loader=lambda _: _Session(),
    )

    assert len(coverage) == 1
    row = coverage.iloc[0]
    assert bool(row["event_in_run_epoch"])
    assert bool(row["past_route_available"])
    assert bool(row["future_route_available"])
    assert bool(row["past_route_informative_10cm"])
    assert bool(row["future_route_informative_10cm"])
    assert float(row["time_to_departure_s"]) == pytest.approx(2.0, abs=0.11)
    assert not any(column.startswith("logZ_") for column in coverage.columns)
    assert "analysis_role" not in coverage


def _write_hc11_session(root: Path, animal: str, session: str) -> None:
    session_dir = root / animal / session
    session_dir.mkdir(parents=True)
    timestamps = np.arange(0.0, 100.0, 0.1)
    x = np.zeros_like(timestamps)
    for peak in (20.0, 50.0):
        moving = (timestamps >= peak + 1.0) & (timestamps <= peak + 5.0)
        x[moving] += (timestamps[moving] - (peak + 1.0)) * 0.10
        x[timestamps > peak + 5.0] += 0.40
    savemat(
        session_dir / f"{session}.position.behavior.mat",
        {
            "position": {
                "timestamps": timestamps,
                "position": {"x": x, "y": np.zeros_like(x), "lin": x},
                "units": "m",
                "Epochs": {"MazeEpoch": np.array([10.0, 90.0])},
                "behaviorinfo": {"MazeType": "synthetic maze"},
            }
        },
    )
    with h5py.File(session_dir / f"{session}.ripplesALL.event.mat", "w") as handle:
        group = handle.create_group("ripples")
        group.create_dataset("peaks", data=np.array([[20.0, 50.0]]))


def test_hc11_external_gate_requires_awake_events_across_animals(tmp_path: Path) -> None:
    for index in range(3):
        _write_hc11_session(tmp_path, f"Rat{index + 1}", f"Session{index + 1}")

    result = audit_hc11_awake_suitability(tmp_path)
    summary = result[result["status"].notna()].iloc[0]

    assert int(summary["animals"]) == 3
    assert int(summary["sessions"]) == 3
    assert int(summary["awake_maze_events"]) == 6
    assert int(summary["events_with_departure_within_window"]) == 6
    assert bool(summary["suitable_for_commitment_confirmation"])


def test_missing_behavior_fails_non_vacuously() -> None:
    evidence = pd.DataFrame(_event_rows("Rat1/Open1", 1, _scores(FIRST_ORDER_IMM)))
    events = build_frozen_event_table(evidence)
    primary = choose_primary_momentum_axis(events)
    external = pd.DataFrame(
        [
            {
                "dataset": "none",
                "status": "not_provided",
                "suitable_for_commitment_confirmation": False,
            }
        ]
    )

    gates = build_gates(
        events,
        primary,
        pd.DataFrame(),
        external,
        minimum_behavior_coverage_fraction=0.9,
    ).set_index("gate")

    assert not bool(gates.loc["behavior_coverage_present", "passed"])
    assert not bool(gates.loc["events_in_run_epoch", "passed"])
    assert not bool(gates.loc["future_route_coverage", "passed"])
    assert not bool(gates.loc["pf_primary_analysis_ready", "passed"])


def test_cli_writes_frozen_non_outcome_pack(tmp_path: Path) -> None:
    evidence = pd.DataFrame(_event_rows("Rat1/Open1", 1, _scores(FIRST_ORDER_IMM)))
    evidence_path = tmp_path / "evidence.csv"
    evidence.to_csv(evidence_path, index=False)
    output = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_replay_commitment_composition_feasibility.py",
            "--event-model-evidence",
            str(evidence_path),
            "--output-dir",
            str(output),
        ],
        check=True,
    )

    assert (output / BEHAVIOR_COVERAGE_OUTPUT).exists()
    assert (output / PRIMARY_AXIS_OUTPUT).exists()
    assert (output / GATE_OUTPUT).exists()
    manifest = json.loads((output / MANIFEST_OUTPUT).read_text())
    assert manifest["outcome_join_performed"] is False
    assert manifest["event_ids_frozen"] is True
