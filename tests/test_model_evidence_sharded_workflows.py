from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_model_evidence_shards import aggregate  # noqa: E402
from plan_model_evidence_event_shards import _event_chunks, _event_spec  # noqa: E402


def test_model_evidence_event_shard_planner_builds_balanced_specs():
    chunks = _event_chunks([10, 11, 12, 20, 21], requested_shards=3)

    assert chunks == [[10], [11, 12], [20, 21]]
    assert [_event_spec(chunk) for chunk in chunks] == ["10", "11-12", "20-21"]


def _write_event_model_evidence(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "event_model_evidence.csv", index=False)


def _row(event_id: int, model: str, family: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_id,
        "model": model,
        "requested_model": model,
        "model_family": family,
        "log_evidence": log_evidence,
        "n_time": 3,
        "n_spikes": 7,
        "runtime_s": 0.1,
        "error": "",
        "bin_size_cm": 6.0,
        "smoothing_sigma_bins": 2.5,
        "min_speed_cm_s": 5.0,
        "time_bin_s": 0.003,
    }


def test_aggregate_model_evidence_shards_recomputes_summary(tmp_path):
    shard_root = tmp_path / "shards"
    out_dir = tmp_path / "out"
    _write_event_model_evidence(
        shard_root / "shard0",
        [
            _row(10, "sorted-spike-state-space-stationary", "nontrajectory", -12.0),
            _row(10, "sorted-spike-state-space-diffusion", "trajectory", -9.0),
            _row(10, "sorted-spike-state-space-momentum", "trajectory", -8.0),
        ],
    )
    _write_event_model_evidence(
        shard_root / "shard1",
        [
            _row(12, "sorted-spike-state-space-stationary", "nontrajectory", -11.0),
            _row(12, "sorted-spike-state-space-diffusion", "trajectory", -7.0),
            _row(12, "sorted-spike-state-space-momentum", "trajectory", -10.0),
        ],
    )

    combined = aggregate(str(shard_root / "**" / "event_model_evidence.csv"), out_dir)
    scores = pd.read_csv(out_dir / "event_model_evidence.csv")
    counts = pd.read_csv(out_dir / "best_model_counts.csv")
    pivot = pd.read_csv(out_dir / "event_model_pivot_log_evidence.csv")

    assert len(combined) == 6
    assert len(scores) == 6
    assert set(scores["event_index"]) == {10, 12}
    assert set(pivot["event_index"]) == {10, 12}
    best = counts[counts["comparison"] == "best_model"].set_index("model")["events"].to_dict()
    assert best == {
        "sorted-spike-state-space-momentum": 1,
        "sorted-spike-state-space-diffusion": 1,
    }
