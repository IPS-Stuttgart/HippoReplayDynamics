from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))

import model_evidence_support_audit as support_audit  # noqa: E402
from hipporeplayimm.evidence_reporting import EXACT_EVIDENCE_SUPPORT  # noqa: E402


def test_support_audit_cli_preserves_nullable_large_event_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = 2**53
    second = first + 1
    scores_csv = tmp_path / "scores.csv"
    output_dir = tmp_path / "audit"
    scores_csv.write_text(
        "session,event_index,model,model_family,status,log_evidence,evidence_support,evidence_comparable\n"
        f"Rat1/Open1,{first},diffusion,trajectory,success,-1.0,{EXACT_EVIDENCE_SUPPORT},True\n"
        f"Rat1/Open1,{second},diffusion,trajectory,success,-2.0,{EXACT_EVIDENCE_SUPPORT},True\n"
        f"Rat1/Open1,,diffusion,trajectory,success,-3.0,{EXACT_EVIDENCE_SUPPORT},True\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_evidence_support_audit.py",
            str(scores_csv),
            "--output",
            str(output_dir),
        ],
    )

    assert support_audit.main() == 0

    audited = pd.read_csv(
        output_dir / "event_evidence_support_audit.csv",
        dtype={"event_index": "string"},
    )
    exact_ids = set(audited["event_index"].dropna())
    assert exact_ids == {str(first), str(second)}
    assert audited["event_index"].isna().sum() == 1
