import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_SCRIPT = _SCRIPTS_DIR / "aggregate_model_evidence_shards.py"
_SPEC = importlib.util.spec_from_file_location("aggregate_model_evidence_shards_integer_keys", _SCRIPT)
assert _SPEC is not None
aggregate_model_evidence_shards = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(aggregate_model_evidence_shards)


def test_shard_reader_preserves_large_nullable_split_identifiers(tmp_path: Path) -> None:
    shard = tmp_path / "event_model_evidence_shard.csv"
    shard.write_text(
        "session,event_index,benchmark_cell_split_index,model,log_evidence,status\n"
        "Rat1/Open1,0,9007199254740992,diffusion,-1.0,success\n"
        "Rat1/Open1,0,9007199254740993,diffusion,-2.0,success\n"
        "Rat1/Open1,1,,diffusion,-3.0,success\n",
        encoding="utf-8",
    )

    frame = aggregate_model_evidence_shards._read_event_score_csv(shard)

    assert frame.loc[0, "benchmark_cell_split_index"] == 2**53
    assert frame.loc[1, "benchmark_cell_split_index"] == 2**53 + 1
    assert frame.loc[0, "benchmark_cell_split_index"] != frame.loc[1, "benchmark_cell_split_index"]
    assert pd.isna(frame.loc[2, "benchmark_cell_split_index"])
