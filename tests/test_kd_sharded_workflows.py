from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_kd_model_evidence_shards import aggregate, _load_momentum_grid  # noqa: E402
from plan_kd_event_shards import _event_chunks, _event_spec  # noqa: E402


def _write_momentum_shard(
    path: Path,
    *,
    event_ids: list[int],
    sd_indices: list[int],
    decay_indices: list[int],
    values: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        session="RatX/OpenY",
        event_ids=np.asarray(event_ids, dtype=int),
        n_time=np.full(len(event_ids), 3, dtype=int),
        n_spikes=np.full(len(event_ids), 7, dtype=int),
        sd_meters=np.asarray([0.1, 0.2], dtype=float),
        decay=np.asarray([1.0, 2.0], dtype=float),
        sd_indices=np.asarray(sd_indices, dtype=int),
        decay_indices=np.asarray(decay_indices, dtype=int),
        values=np.asarray(values, dtype=float),
        runtime_s=np.asarray(10.0, dtype=float),
        shard_index=np.asarray(0, dtype=int),
        shard_count=np.asarray(2, dtype=int),
        kd_grid_preset="smoke",
        kd_time_bin_ms=np.asarray(3.0, dtype=float),
        kd_bin_size_cm=np.asarray(4.0, dtype=float),
        kd_n_bins=np.asarray(2, dtype=int),
        kd_n_jobs=np.asarray(1, dtype=int),
        kd_event_chunk_size=np.asarray(1, dtype=int),
    )


def test_event_shard_planner_builds_balanced_nonempty_specs():
    chunks = _event_chunks([10, 11, 12, 20, 21], requested_shards=3)

    assert chunks == [[10], [11, 12], [20, 21]]
    assert [_event_spec(chunk) for chunk in chunks] == ["10", "11-12", "20-21"]


def test_momentum_grid_loader_rejects_negative_sd_indices(tmp_path):
    shard = tmp_path / "negative_sd_index.npz"
    _write_momentum_shard(
        shard,
        event_ids=[10],
        sd_indices=[-1, 0, 0, 1],
        decay_indices=[0, 0, 1, 1],
        values=np.array([[-7.0, -8.0, -9.0, -10.0]]),
    )

    with pytest.raises(ValueError, match="sd_indices out of range"):
        _load_momentum_grid([shard])


def test_momentum_grid_loader_rejects_out_of_range_decay_indices(tmp_path):
    shard = tmp_path / "bad_decay_index.npz"
    _write_momentum_shard(
        shard,
        event_ids=[10],
        sd_indices=[0, 1, 0, 1],
        decay_indices=[0, 0, 1, 2],
        values=np.array([[-7.0, -8.0, -9.0, -10.0]]),
    )

    with pytest.raises(ValueError, match="decay_indices out of range"):
        _load_momentum_grid([shard])


def test_aggregate_accepts_distinct_event_and_grid_shards(tmp_path):
    base_dir = tmp_path / "base"
    shards_dir = tmp_path / "shards"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    shards_dir.mkdir()
    events = [10, 12]
    base_rows = []
    for event_id in events:
        for model, family, log_evidence in (
            ("random", "nontrajectory", -10.0 - event_id),
            ("stationary", "nontrajectory", -11.0 - event_id),
            ("stationary-gaussian", "nontrajectory", -9.0 - event_id),
            ("diffusion", "trajectory", -8.0 - event_id),
        ):
            base_rows.append(
                {
                    "status": "success",
                    "session": "RatX/OpenY",
                    "event_index": event_id,
                    "model": model,
                    "model_family": family,
                    "log_evidence": log_evidence,
                    "n_time": 3,
                    "n_spikes": 7,
                    "runtime_s": 0.1,
                    "error": "",
                    "kd_grid_preset": "smoke",
                    "kd_time_bin_ms": 3.0,
                    "kd_bin_size_cm": 4.0,
                    "kd_n_bins": 2,
                    "kd_n_jobs": 1,
                    "kd_event_chunk_size": 1,
                }
            )
    pd.DataFrame(base_rows).to_csv(base_dir / "event_model_evidence.csv", index=False)
    pd.DataFrame(
        [
            {"event_index": event_id, "model": "diffusion", "best_sd_meters": 0.1, "best_log_evidence": -8.0 - event_id}
            for event_id in events
        ]
    ).to_csv(base_dir / "gridsearch_best_params.csv", index=False)
    pd.DataFrame(
        [
            {"session": "RatX/OpenY", "event_index": event_id, "model": "diffusion", "log_evidence": -8.0 - event_id}
            for event_id in events
        ]
    ).to_csv(base_dir / "marginalized_model_evidence.csv", index=False)

    _write_momentum_shard(
        shards_dir / "event0_grid0.npz",
        event_ids=[10],
        sd_indices=[0, 1],
        decay_indices=[0, 0],
        values=np.array([[-7.0, -9.0]]),
    )
    _write_momentum_shard(
        shards_dir / "event0_grid1.npz",
        event_ids=[10],
        sd_indices=[0, 1],
        decay_indices=[1, 1],
        values=np.array([[-6.0, -10.0]]),
    )
    _write_momentum_shard(
        shards_dir / "event1_grid0.npz",
        event_ids=[12],
        sd_indices=[0, 1],
        decay_indices=[0, 0],
        values=np.array([[-20.0, -19.0]]),
    )
    _write_momentum_shard(
        shards_dir / "event1_grid1.npz",
        event_ids=[12],
        sd_indices=[0, 1],
        decay_indices=[1, 1],
        values=np.array([[-18.0, -17.0]]),
    )

    aggregate(base_dir, str(shards_dir / "*.npz"), out_dir)

    scores = pd.read_csv(out_dir / "event_model_evidence.csv")
    pivot = pd.read_csv(out_dir / "event_model_pivot_log_evidence.csv")

    assert len(scores) == 10
    assert set(scores["event_index"]) == {10, 12}
    assert set(scores["model"]) == {"random", "stationary", "stationary-gaussian", "diffusion", "momentum"}
    assert pivot["momentum"].notna().all()
