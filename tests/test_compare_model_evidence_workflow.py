from pathlib import Path


def test_compare_model_evidence_runs_workflow_downloads_two_artifacts_and_uploads_tables():
    workflow = Path(".github/workflows/compare-model-evidence-runs.yml").read_text(encoding="utf-8")

    assert "name: Compare model-evidence runs" in workflow
    assert "left_run_id:" in workflow
    assert "right_run_id:" in workflow
    assert "left_artifact_name:" in workflow
    assert "right_artifact_name:" in workflow
    assert "actions/download-artifact@v7" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "actions/upload-artifact@v4" not in workflow
    assert "scripts/compare_model_evidence_artifacts.py" in workflow
    assert "--exact-only" in workflow
    assert "--left artifacts/left" in workflow
    assert "--right artifacts/right" in workflow
    assert "event_best_model_comparison.csv" in workflow
    assert "canonical_best_model_crosstab.csv" in workflow
    assert "evidence_support_counts.csv" in workflow
    assert "shared_relative_evidence_summary.csv" in workflow
    assert "session_story_shift_summary.csv" in workflow
    assert "model-evidence-run-comparison-" in workflow
