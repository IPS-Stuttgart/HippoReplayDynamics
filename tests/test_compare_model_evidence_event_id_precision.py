from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_model_evidence_artifacts.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("compare_model_evidence_event_id_precision", _SCRIPT)
assert _SPEC is not None
compare_model_evidence_artifacts = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_model_evidence_event_id_precision"] = compare_model_evidence_artifacts
assert _SPEC.loader is not None
_SPEC.loader.exec_module(compare_model_evidence_artifacts)


def test_read_score_csv_preserves_nullable_large_event_ids(tmp_path: Path) -> None:
    first = 2**53
    second = first + 1
    scores_csv = tmp_path / "scores.csv"
    scores_csv.write_text(
        "session,event_index,event_id,model,log_evidence\n"
        f"Rat1/Open1,0,{first},diffusion,-1.0\n"
        f"Rat1/Open1,1,{second},diffusion,-2.0\n"
        "Rat1/Open1,2,,diffusion,-3.0\n",
        encoding="utf-8",
    )

    scores = compare_model_evidence_artifacts._read_score_csv(scores_csv)

    assert scores["event_id"].iloc[0] == str(first)
    assert scores["event_id"].iloc[1] == str(second)
    assert scores["event_id"].iloc[0] != scores["event_id"].iloc[1]
    assert pd.isna(scores["event_id"].iloc[2])
