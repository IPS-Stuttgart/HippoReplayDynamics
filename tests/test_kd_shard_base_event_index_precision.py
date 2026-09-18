from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_kd_model_evidence_shards import (  # noqa: E402
    _read_csv_with_exact_event_index,
    aggregate,
)


def _write_momentum_shard(path: Path, event_id: int) -> None:
    np.savez_compressed(
        path,
        session="RatX/OpenY",
        event_ids=np.asarray([event_id], dtype=np.int64),
        n_time=np.asarray([3], dtype=int),
        n_spikes=np.asarray([7], dtype=int),
        sd_meters=np.asarray([0.1], dtype=float),
        decay=np.asarray([1.0], dtype=float),
        sd_indices=np.asarray([0], dtype=int),
        decay_indices=np.asarray([0], dtype=int),
        values=np.asarray([[-7.0]], dtype=float),
        runtime_s=np.asarray(0.1, dtype=float),
        kd_grid_preset="smoke",
        kd_time_bin_ms=np.asarray(3.0, dtype=float),
        kd_bin_size_cm=np.asarray(4.0, dtype=float),
        kd_n_bins=np.asarray(2, dtype=int),
        kd_n_jobs=np.asarray(1, dtype=int),
        kd_event_chunk_size=np.asarray(1, dtype=int),
    )


def test_base_csv_reader_preserves_adjacent_decimal_event_ids(tmp_path: Path) -> None:
    first = 2**53
    second = first + 1
    if np.iinfo(np.intp).max < second:
        pytest.skip("platform integer type cannot hold the regression identifiers")
    path = tmp_path / "event_model_evidence.csv"
    pd.DataFrame(
        {
            "event_index": [f"{first}.0", f"{second}.0"],
            "model": ["diffusion", "diffusion"],
        }
    ).to_csv(path, index=False)

    loaded = _read_csv_with_exact_event_index(path)

    assert loaded["event_index"].tolist() == [first, second]


def test_base_csv_reader_rejects_fractional_event_ids(tmp_path: Path) -> None:
    path = tmp_path / "event_model_evidence.csv"
    pd.DataFrame({"event_index": ["10.5"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="finite integer values"):
        _read_csv_with_exact_event_index(path)


def test_aggregate_preserves_adjacent_decimal_event_ids_above_binary64_precision(
    tmp_path: Path,
) -> None:
    first = 2**53
    second = first + 1
    if np.iinfo(np.intp).max < second:
        pytest.skip("platform integer type cannot hold the regression identifiers")

    base_dir = tmp_path / "base"
    shards_dir = tmp_path / "shards"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    shards_dir.mkdir()

    base_rows: list[dict[str, object]] = []
    for event_id in (first, second):
        event_text = f"{event_id}.0"
        for model, family, log_evidence in (
            ("random", "nontrajectory", -10.0),
            ("stationary", "nontrajectory", -11.0),
            ("stationary-gaussian", "nontrajectory", -9.0),
            ("diffusion", "trajectory", -8.0),
        ):
            base_rows.append(
                {
                    "status": "success",
                    "session": "RatX/OpenY",
                    "event_index": event_text,
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
            {
                "event_index": f"{event_id}.0",
                "model": "diffusion",
                "best_sd_meters": 0.1,
                "best_log_evidence": -8.0,
            }
            for event_id in (first, second)
        ]
    ).to_csv(base_dir / "gridsearch_best_params.csv", index=False)
    pd.DataFrame(
        [
            {
                "session": "RatX/OpenY",
                "event_index": f"{event_id}.0",
                "model": "diffusion",
                "log_evidence": -8.0,
            }
            for event_id in (first, second)
        ]
    ).to_csv(base_dir / "marginalized_model_evidence.csv", index=False)

    _write_momentum_shard(shards_dir / "event0.npz", first)
    _write_momentum_shard(shards_dir / "event1.npz", second)

    aggregate(base_dir, str(shards_dir / "*.npz"), out_dir)

    scores = pd.read_csv(
        out_dir / "event_model_evidence.csv",
        dtype={"event_index": "string"},
    )
    observed = {int(value) for value in scores["event_index"].drop_duplicates()}

    assert observed == {first, second}
