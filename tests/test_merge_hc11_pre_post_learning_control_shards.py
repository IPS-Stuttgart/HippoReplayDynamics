from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_hc11_pre_post_learning_controls as audit  # noqa: E402
import merge_hc11_pre_post_learning_control_shards as merge  # noqa: E402


def _write_shard(path: Path, session: str) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "session": session,
                "phase": "PRE",
                "event_id": 1,
                "population": "all",
                "control_type": "original",
                "replicate": 0,
            }
        ]
    ).to_csv(path / audit.CONTROL_EVIDENCE_OUTPUT, index=False)
    pd.DataFrame(
        [
            {
                "session": session,
                "phase": "PRE",
                "event_id": 1,
                "population": "all",
                "split_index": 0,
            }
        ]
    ).to_csv(path / audit.HELDOUT_OUTPUT, index=False)


def test_shard_loader_rejects_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_shard(first, "RatA_day1")
    _write_shard(second, "RatB_day1")
    controls, heldout = merge.load_control_shards([first, second])
    assert len(controls) == 2
    assert len(heldout) == 2

    duplicate = tmp_path / "duplicate"
    _write_shard(duplicate, "RatA_day1")
    try:
        merge.load_control_shards([first, duplicate])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate control shards should fail")
