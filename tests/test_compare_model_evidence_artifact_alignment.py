from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_model_evidence_artifacts.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("compare_model_evidence_artifacts_alignment", _SCRIPT)
assert _SPEC is not None
compare_model_evidence_artifacts = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_model_evidence_artifacts_alignment"] = compare_model_evidence_artifacts
assert _SPEC.loader is not None
_SPEC.loader.exec_module(compare_model_evidence_artifacts)


def _score_row(model: str, log_evidence: float, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "session": "Rat1/Open1",
        "event_index": 7,
        "model": model,
        "log_evidence": log_evidence,
    }
    row.update(extra)
    return row


def _write_scores(root: Path, filename: str, rows: list[dict[str, object]]) -> None:
    root.mkdir()
    pd.DataFrame(rows).to_csv(root / filename, index=False)


def test_compare_artifacts_rejects_ambiguous_alignment_when_discriminator_is_missing(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_scores(
        left,
        "event_model_evidence.csv",
        [
            _score_row("diffusion", -10.0, null_random_seed=11),
            _score_row("momentum", -1.0, null_random_seed=11),
            _score_row("diffusion", -2.0, null_random_seed=12),
            _score_row("momentum", -9.0, null_random_seed=12),
        ],
    )
    _write_scores(
        right,
        "all_sessions_event_model_evidence.csv",
        [
            _score_row("sorted-spike-state-space-diffusion", -11.0),
            _score_row("sorted-spike-state-space-momentum", -2.0),
        ],
    )

    with pytest.raises(ValueError, match="cannot be aligned uniquely"):
        compare_model_evidence_artifacts.compare_artifacts(
            left,
            right,
            left_label="left",
            right_label="right",
            output=tmp_path / "comparison",
        )
