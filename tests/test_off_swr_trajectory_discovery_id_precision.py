from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from off_swr_trajectory_discovery import (  # noqa: E402
    _read_score_files,
    cluster_off_swr_candidates,
    off_swr_trajectory_candidates,
    off_swr_trajectory_decisions,
)


MODELS = (
    ("sorted-spike-state-space-stationary", 0.0),
    ("sorted-spike-state-space-diffusion", 7.0),
    ("sorted-spike-state-space-fragmented", 8.0),
    ("sorted-spike-state-space-first-order-imm", 10.0),
    ("sorted-spike-state-space-momentum-exact-sparse", 9.0),
)


def _event_rows(event_index: object, *, start: float) -> list[dict[str, object]]:
    return [
        {
            "status": "success",
            "session": "Rat1/Open1",
            "event_index": event_index,
            "window_role": "matched_null",
            "null_index": "0.0",
            "model": model,
            "log_evidence": log_evidence,
            "evidence_comparable": True,
            "off_swr": True,
            "window_start_s": start,
            "window_end_s": start + 0.1,
            "window_duration_s": 0.1,
            "n_spikes": 10,
            "n_time": 20,
            "null_active_cell_count": 5,
        }
        for model, log_evidence in MODELS
    ]


def test_off_swr_discovery_preserves_adjacent_decimal_event_ids_from_csv(
    tmp_path: Path,
) -> None:
    first = 2**53
    second = first + 1
    path = tmp_path / "matched_null_event_model_evidence.csv"
    pd.DataFrame(
        [
            *_event_rows(f"{first}.0", start=10.0),
            *_event_rows(f"{second}.0", start=10.2),
        ]
    ).to_csv(path, index=False)

    scores = _read_score_files(path)
    decisions = off_swr_trajectory_decisions(scores)
    candidates = off_swr_trajectory_candidates(decisions)
    clusters = cluster_off_swr_candidates(candidates, cluster_gap_s=0.5)

    assert scores["event_index"].drop_duplicates().tolist() == [first, second]
    assert decisions["event_index"].tolist() == [first, second]
    assert len(candidates) == 2
    assert len(clusters) == 1
    assert clusters.iloc[0]["template_event_indices"] == f"{first} {second}"


def test_off_swr_discovery_rejects_ambiguous_large_float_event_id() -> None:
    scores = pd.DataFrame(_event_rows(float(2**53), start=10.0))

    with pytest.raises(ValueError, match="outside the exact integer range"):
        off_swr_trajectory_decisions(scores)
