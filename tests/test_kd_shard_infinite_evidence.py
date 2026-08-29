from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_kd_model_evidence_shards import _load_momentum_grid  # noqa: E402


def _write_shard(path: Path, value: float) -> None:
    np.savez(
        path,
        session=np.asarray("rat-test"),
        event_ids=np.asarray([7], dtype=np.int64),
        n_time=np.asarray([2], dtype=np.int64),
        n_spikes=np.asarray([3], dtype=np.int64),
        sd_meters=np.asarray([0.1], dtype=float),
        decay=np.asarray([0.9], dtype=float),
        sd_indices=np.asarray([0], dtype=np.int64),
        decay_indices=np.asarray([0], dtype=np.int64),
        values=np.asarray([[value]], dtype=float),
        runtime_s=np.asarray(0.2),
        kd_grid_preset=np.asarray("test"),
        kd_time_bin_ms=np.asarray(20.0),
        kd_bin_size_cm=np.asarray(4.0),
        kd_n_bins=np.asarray(1, dtype=np.int64),
        kd_n_jobs=np.asarray(1, dtype=np.int64),
        kd_event_chunk_size=np.asarray(1, dtype=np.int64),
        kd_spike_rate_scale=np.asarray(1.0),
    )


def test_load_momentum_grid_accepts_negative_infinite_log_evidence(tmp_path: Path) -> None:
    shard = tmp_path / "negative_infinity.npz"
    _write_shard(shard, float("-inf"))

    grid, params, metadata = _load_momentum_grid([shard])

    assert grid.shape == (1, 1, 1)
    assert np.isneginf(grid[0, 0, 0])
    assert params["sd_meters"].tolist() == [0.1]
    assert metadata["event_ids"].tolist() == [7]


def test_duplicate_negative_infinite_log_evidence_is_still_a_duplicate(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_shard(first, float("-inf"))
    _write_shard(second, float("-inf"))

    with pytest.raises(ValueError, match="Duplicate momentum score"):
        _load_momentum_grid([first, second])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_load_momentum_grid_rejects_invalid_nonfinite_values(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    shard = tmp_path / "invalid.npz"
    _write_shard(shard, invalid_value)

    with pytest.raises(ValueError, match="finite or -inf log evidence"):
        _load_momentum_grid([shard])
