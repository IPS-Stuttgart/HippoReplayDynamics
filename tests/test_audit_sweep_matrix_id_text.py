from __future__ import annotations

import pandas as pd

from scripts.audit_sweep_completeness import audit_sweep_completeness


def test_sweep_completeness_preserves_leading_zero_matrix_ids(tmp_path) -> None:
    root = tmp_path / "artifacts"
    plan = root / "state-space-evidence-sweep-plan-1"
    plan.mkdir(parents=True)
    pd.DataFrame([{"id": "001"}]).to_csv(plan / "matrix.csv", index=False)

    run = root / "state-space-evidence-sweep-001"
    run.mkdir()
    pd.DataFrame(
        [{"status": "success", "session": "Rat1/Open1", "event_index": 0}]
    ).to_csv(run / "event_model_evidence.csv", index=False)
    pd.DataFrame([{"model": "overall"}]).to_csv(
        run / "model_evidence_summary.csv",
        index=False,
    )

    table = audit_sweep_completeness(
        artifact_root=root,
        output=tmp_path / "completeness.csv",
        mode="state-space-evidence",
    )

    assert table["matrix_id"].tolist() == ["001"]
    assert bool(table.loc[0, "planned"])
    assert bool(table.loc[0, "artifact_complete"])
    assert bool(table.loc[0, "included_in_final_ranking"])


def test_sweep_completeness_preserves_csv_matrix_id_spelling(tmp_path) -> None:
    root = tmp_path / "artifacts"
    plan = root / "state-space-evidence-sweep-plan-1"
    plan.mkdir(parents=True)
    pd.DataFrame([{"id": "001"}]).to_csv(plan / "matrix.csv", index=False)

    run = root / "state-space-evidence-sweep-unrelated-name"
    run.mkdir()
    pd.DataFrame(
        [{"status": "success", "matrix_id": "001", "event_index": 0}]
    ).to_csv(run / "event_model_evidence.csv", index=False)
    pd.DataFrame([{"matrix_id": "001", "model": "overall"}]).to_csv(
        run / "model_evidence_summary.csv",
        index=False,
    )

    table = audit_sweep_completeness(
        artifact_root=root,
        output=tmp_path / "completeness.csv",
        mode="state-space-evidence",
    )

    assert table["matrix_id"].tolist() == ["001"]
    assert bool(table.loc[0, "artifact_complete"])
