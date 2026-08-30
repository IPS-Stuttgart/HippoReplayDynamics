from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_trajectory_family_paper_pack import (
    FULL_CORE_EVENT_TABLE_CANDIDATES,
    _resolve_csv,
)


def test_resolve_csv_rejects_ambiguous_nested_full_core_tables(tmp_path: Path) -> None:
    candidate_name = FULL_CORE_EVENT_TABLE_CANDIDATES[0]
    for run_name in ("run-b", "run-a"):
        table = tmp_path / run_name / candidate_name
        table.parent.mkdir()
        table.write_text("session,event_index,model,log_evidence\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"Multiple {candidate_name} files found"):
        _resolve_csv(tmp_path, FULL_CORE_EVENT_TABLE_CANDIDATES)
