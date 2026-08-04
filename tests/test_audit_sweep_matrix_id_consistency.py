from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit_sweep_completeness import audit_sweep_completeness


def test_sweep_completeness_rejects_mixed_matrix_ids_in_one_artifact(tmp_path) -> None:
    root = tmp_path / "artifacts"
    plan = root / "state-space-evidence-sweep-plan-1"
    plan.mkdir(parents=True)
    pd.DataFrame([{"id": "a"}, {"id": "b"}]).to_csv(
        plan / "matrix.csv",
        index=False,
    )

    run = root / "state-space-evidence-sweep-a"
    run.mkdir()
    pd.DataFrame(
        [
            {"status": "success", "matrix_id": "a", "event_index": 0},
            {"status": "success", "matrix_id": "b", "event_index": 1},
        ]
    ).to_csv(run / "event_model_evidence.csv", index=False)
    pd.DataFrame([{"matrix_id": "a", "model": "overall"}]).to_csv(
        run / "model_evidence_summary.csv",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=r"event_model_evidence\.csv contains multiple matrix IDs.*'a'.*'b'",
    ):
        audit_sweep_completeness(
            artifact_root=root,
            output=tmp_path / "completeness.csv",
            mode="state-space-evidence",
        )
