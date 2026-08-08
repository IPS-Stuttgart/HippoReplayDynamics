from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_model_evidence_shards.py"
_SCRIPT_DIRECTORY = str(_SCRIPT.parent)
_SPEC = importlib.util.spec_from_file_location("aggregate_model_evidence_shards_output_exclusion", _SCRIPT)
assert _SPEC is not None
aggregate_model_evidence_shards = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.path.insert(0, _SCRIPT_DIRECTORY)
try:
    _SPEC.loader.exec_module(aggregate_model_evidence_shards)
finally:
    sys.path.remove(_SCRIPT_DIRECTORY)


def test_recursive_glob_excludes_prior_aggregate_output(tmp_path):
    shard = tmp_path / "results" / "shards" / "shard0" / "event_model_evidence.csv"
    shard.parent.mkdir(parents=True)
    shard.write_text("model,log_evidence\nstationary,0\n", encoding="utf-8")

    outdir = tmp_path / "results" / "model-evidence"
    outdir.mkdir(parents=True)
    prior_aggregate = outdir / "event_model_evidence.csv"
    prior_aggregate.write_text("model,log_evidence\nstationary,0\n", encoding="utf-8")

    paths = aggregate_model_evidence_shards._load_score_files(
        str(tmp_path / "results" / "**" / "event_model_evidence.csv")
    )
    filtered = aggregate_model_evidence_shards._exclude_aggregate_output(paths, outdir)

    assert filtered == [shard]
