from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _write_shard(path: Path, *, decay_index: int, values: np.ndarray, runtime_s: float) -> None:
    np.savez(
        path,
        session=np.asarray("Rat1/Open1"),
        event_ids=np.asarray([0, 1], dtype=int),
        n_time=np.asarray([3, 4], dtype=int),
        n_spikes=np.asarray([5, 6], dtype=int),
        sd_meters=np.asarray([0.05], dtype=float),
        decay=np.asarray([0.93, 0.97], dtype=float),
        sd_indices=np.asarray([0], dtype=int),
        decay_indices=np.asarray([decay_index], dtype=int),
        values=np.asarray(values, dtype=float),
        runtime_s=np.asarray(runtime_s, dtype=float),
        kd_grid_preset=np.asarray("tiny"),
        kd_time_bin_ms=np.asarray(4.0, dtype=float),
        kd_bin_size_cm=np.asarray(4.0, dtype=float),
        kd_n_bins=np.asarray(8, dtype=int),
        kd_n_jobs=np.asarray(1, dtype=int),
        kd_event_chunk_size=np.asarray(2, dtype=int),
        kd_spike_rate_scale=np.asarray(1.0, dtype=float),
    )


def test_momentum_shard_runtime_accumulates_across_grid_chunks(tmp_path: Path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    aggregator = importlib.import_module("aggregate_kd_model_evidence_shards")
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    _write_shard(left, decay_index=0, values=np.asarray([[1.0], [2.0]]), runtime_s=4.0)
    _write_shard(right, decay_index=1, values=np.asarray([[3.0], [4.0]]), runtime_s=6.0)

    _, _, metadata = aggregator._load_momentum_grid([left, right])

    # Each shard covers both events, so the per-event runtime is half of each
    # shard runtime and should add over grid chunks: 4/2 + 6/2 = 5 seconds.
    np.testing.assert_allclose(
        metadata["runtime_s_by_event"],
        np.asarray([5.0, 5.0], dtype=float),
    )
