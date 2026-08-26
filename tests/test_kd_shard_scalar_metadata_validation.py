from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_kd_model_evidence_shards import _load_momentum_grid  # noqa: E402


def _write_shard(path: Path, **metadata_overrides: object) -> None:
    metadata: dict[str, object] = {
        "kd_n_bins": np.asarray(1, dtype=int),
        "kd_n_jobs": np.asarray(1, dtype=int),
        "kd_event_chunk_size": np.asarray(1, dtype=int),
    }
    metadata.update(metadata_overrides)
    np.savez_compressed(
        path,
        session="RatX/OpenY",
        event_ids=np.asarray([10], dtype=int),
        n_time=np.asarray([3], dtype=int),
        n_spikes=np.asarray([7], dtype=int),
        sd_meters=np.asarray([0.1], dtype=float),
        decay=np.asarray([1.0], dtype=float),
        sd_indices=np.asarray([0], dtype=int),
        decay_indices=np.asarray([0], dtype=int),
        values=np.asarray([[-7.0]], dtype=float),
        runtime_s=np.asarray(10.0, dtype=float),
        shard_index=np.asarray(0, dtype=int),
        shard_count=np.asarray(1, dtype=int),
        kd_grid_preset="smoke",
        kd_time_bin_ms=np.asarray(3.0, dtype=float),
        kd_bin_size_cm=np.asarray(4.0, dtype=float),
        **metadata,
    )


@pytest.mark.parametrize("key", ["kd_n_bins", "kd_n_jobs", "kd_event_chunk_size"])
def test_momentum_grid_loader_rejects_fractional_scalar_integer_metadata(
    tmp_path: Path,
    key: str,
) -> None:
    shard = tmp_path / f"fractional_{key}.npz"
    _write_shard(shard, **{key: np.asarray(1.5, dtype=float)})

    with pytest.raises(ValueError, match=rf"{key}.*finite integer values"):
        _load_momentum_grid([shard])


@pytest.mark.parametrize("key", ["kd_n_bins", "kd_n_jobs", "kd_event_chunk_size"])
def test_momentum_grid_loader_rejects_nonpositive_scalar_integer_metadata(
    tmp_path: Path,
    key: str,
) -> None:
    shard = tmp_path / f"zero_{key}.npz"
    _write_shard(shard, **{key: np.asarray(0, dtype=int)})

    with pytest.raises(ValueError, match=rf"{key}.*positive integer values"):
        _load_momentum_grid([shard])


def test_momentum_grid_loader_rejects_boolean_scalar_integer_metadata(tmp_path: Path) -> None:
    shard = tmp_path / "boolean_kd_n_jobs.npz"
    _write_shard(shard, kd_n_jobs=np.asarray(True, dtype=bool))

    with pytest.raises(TypeError, match=r"kd_n_jobs.*booleans"):
        _load_momentum_grid([shard])
