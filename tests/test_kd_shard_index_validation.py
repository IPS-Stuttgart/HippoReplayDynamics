from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_kd_model_evidence_shards import _load_momentum_grid  # noqa: E402


def _write_float_index_shard(path: Path) -> None:
    np.savez_compressed(
        path,
        session="RatX/OpenY",
        event_ids=np.asarray([10], dtype=int),
        n_time=np.asarray([3], dtype=int),
        n_spikes=np.asarray([7], dtype=int),
        sd_meters=np.asarray([0.1, 0.2], dtype=float),
        decay=np.asarray([1.0, 2.0], dtype=float),
        sd_indices=np.asarray([0.0, 1.5, 0.0, 1.0], dtype=float),
        decay_indices=np.asarray([0, 0, 1, 1], dtype=int),
        values=np.asarray([[-7.0, -8.0, -9.0, -10.0]], dtype=float),
        runtime_s=np.asarray(10.0, dtype=float),
        shard_index=np.asarray(0, dtype=int),
        shard_count=np.asarray(1, dtype=int),
        kd_grid_preset="smoke",
        kd_time_bin_ms=np.asarray(3.0, dtype=float),
        kd_bin_size_cm=np.asarray(4.0, dtype=float),
        kd_n_bins=np.asarray(2, dtype=int),
        kd_n_jobs=np.asarray(1, dtype=int),
        kd_event_chunk_size=np.asarray(1, dtype=int),
    )


def _write_overlarge_index_shard(path: Path) -> None:
    too_large = float(np.iinfo(np.dtype(np.intp)).max) * 2.0
    np.savez_compressed(
        path,
        session="RatX/OpenY",
        event_ids=np.asarray([10], dtype=int),
        n_time=np.asarray([3], dtype=int),
        n_spikes=np.asarray([7], dtype=int),
        sd_meters=np.asarray([0.1], dtype=float),
        decay=np.asarray([1.0], dtype=float),
        sd_indices=np.asarray([too_large], dtype=float),
        decay_indices=np.asarray([0], dtype=int),
        values=np.asarray([[-7.0]], dtype=float),
        runtime_s=np.asarray(10.0, dtype=float),
        shard_index=np.asarray(0, dtype=int),
        shard_count=np.asarray(1, dtype=int),
        kd_grid_preset="smoke",
        kd_time_bin_ms=np.asarray(3.0, dtype=float),
        kd_bin_size_cm=np.asarray(4.0, dtype=float),
        kd_n_bins=np.asarray(1, dtype=int),
        kd_n_jobs=np.asarray(1, dtype=int),
        kd_event_chunk_size=np.asarray(1, dtype=int),
    )


def _write_overlarge_integer_index_shard(path: Path) -> None:
    too_large = np.uint64(np.iinfo(np.dtype(np.intp)).max) + np.uint64(1)
    np.savez_compressed(
        path,
        session="RatX/OpenY",
        event_ids=np.asarray([10], dtype=int),
        n_time=np.asarray([3], dtype=int),
        n_spikes=np.asarray([7], dtype=int),
        sd_meters=np.asarray([0.1], dtype=float),
        decay=np.asarray([1.0], dtype=float),
        sd_indices=np.asarray([too_large], dtype=np.uint64),
        decay_indices=np.asarray([0], dtype=int),
        values=np.asarray([[-7.0]], dtype=float),
        runtime_s=np.asarray(10.0, dtype=float),
        shard_index=np.asarray(0, dtype=int),
        shard_count=np.asarray(1, dtype=int),
        kd_grid_preset="smoke",
        kd_time_bin_ms=np.asarray(3.0, dtype=float),
        kd_bin_size_cm=np.asarray(4.0, dtype=float),
        kd_n_bins=np.asarray(1, dtype=int),
        kd_n_jobs=np.asarray(1, dtype=int),
        kd_event_chunk_size=np.asarray(1, dtype=int),
    )


def test_momentum_grid_loader_rejects_fractional_grid_indices(tmp_path):
    shard = tmp_path / "fractional_index.npz"
    _write_float_index_shard(shard)

    with pytest.raises(ValueError, match="sd_indices.*integer grid indices"):
        _load_momentum_grid([shard])


def test_momentum_grid_loader_rejects_overlarge_float_grid_indices(tmp_path):
    shard = tmp_path / "overlarge_index.npz"
    _write_overlarge_index_shard(shard)

    with pytest.raises(ValueError, match="sd_indices.*integer index range"):
        _load_momentum_grid([shard])


def test_momentum_grid_loader_rejects_overlarge_integer_grid_indices(tmp_path):
    shard = tmp_path / "overlarge_integer_index.npz"
    _write_overlarge_integer_index_shard(shard)

    with pytest.raises(ValueError, match="sd_indices.*integer index range"):
        _load_momentum_grid([shard])
