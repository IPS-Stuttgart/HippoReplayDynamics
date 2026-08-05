from __future__ import annotations

import threading

import numpy as np
import pytest

import hipporeplayimm.duration_occupancy as duration_occupancy
import hipporeplayimm.state_space as state_space
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, StateSpaceReplayModel


def _emissions(transition_duration: float) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((2, 2), dtype=float),
        spike_counts=np.zeros((2, 0), dtype=int),
        times=np.asarray([0.0, transition_duration], dtype=float),
        dt=0.02,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
        transition_durations=np.asarray([transition_duration], dtype=float),
    )


def test_concurrent_scores_keep_first_order_imm_durations_isolated(monkeypatch) -> None:
    first_waiting = threading.Event()
    release_first = threading.Event()
    other_waiting = threading.Event()
    release_other = threading.Event()

    original_valid_bin_mask = state_space._valid_bin_mask_from_occupancy
    original_fragmented = state_space._score_fragmented

    def blocking_valid_bin_mask(*args, **kwargs):
        if threading.current_thread().name == "first-order-imm":
            first_waiting.set()
            if not release_first.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to resume first-order IMM score")
        return original_valid_bin_mask(*args, **kwargs)

    def blocking_fragmented(*args, **kwargs):
        if threading.current_thread().name == "other-model":
            other_waiting.set()
            if not release_other.wait(timeout=5.0):
                raise RuntimeError("timed out waiting to resume other model score")
        return original_fragmented(*args, **kwargs)

    trajectory = np.asarray(
        [
            [0.0, -1.0e300],
            [-1.0e300, 0.0],
        ],
        dtype=float,
    )
    mode_posterior = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    monkeypatch.setattr(state_space, "_valid_bin_mask_from_occupancy", blocking_valid_bin_mask)
    monkeypatch.setattr(state_space, "_score_fragmented", blocking_fragmented)
    monkeypatch.setattr(
        duration_occupancy,
        "_score_first_order_imm_variable",
        lambda *args, **kwargs: (0.0, trajectory, mode_posterior),
    )

    first_model = StateSpaceReplayModel(
        mode="first-order-imm",
        config=StateSpaceDecoderConfig(mode="first-order-imm"),
    )
    other_model = StateSpaceReplayModel(
        mode="fragmented",
        config=StateSpaceDecoderConfig(mode="fragmented"),
    )
    bin_centers = np.asarray([[0.0], [1.0]], dtype=float)
    results: dict[str, object] = {}
    errors: dict[str, BaseException] = {}

    def run(name: str, model: StateSpaceReplayModel, emissions: LogEmissionTensor) -> None:
        try:
            results[name] = model.score(emissions, bin_centers)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors[name] = exc

    first_thread = threading.Thread(
        target=run,
        args=("first", first_model, _emissions(0.5)),
        name="first-order-imm",
    )
    other_thread = threading.Thread(
        target=run,
        args=("other", other_model, _emissions(0.02)),
        name="other-model",
    )

    try:
        first_thread.start()
        assert first_waiting.wait(timeout=5.0)
        other_thread.start()
        assert other_waiting.wait(timeout=5.0)

        release_first.set()
        first_thread.join(timeout=5.0)
        assert not first_thread.is_alive()
    finally:
        release_first.set()
        release_other.set()
        first_thread.join(timeout=5.0)
        other_thread.join(timeout=5.0)

    assert not errors
    first_score = results["first"]
    assert first_score.diagnostics["state_space_imm_posterior_expected_path_length_cm"] == pytest.approx(1.0)
    assert first_score.diagnostics["state_space_imm_posterior_path_speed_cm_s"] == pytest.approx(2.0)
