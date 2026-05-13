from pathlib import Path


def test_select_state_space_parameters_workflow_downloads_summaries_and_uploads_decision_tables():
    workflow = Path(".github/workflows/select-state-space-parameters.yml").read_text(encoding="utf-8")

    assert "name: Select state-space replay parameters" in workflow
    assert "evidence_run_id:" in workflow
    assert "recovery_run_id:" in workflow
    assert "state-space-evidence-sweep-summary-<run_id>" in workflow
    assert "simulation-recovery-sweep-summary-<run_id>" in workflow
    assert "actions/download-artifact@v7" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "actions/upload-artifact@v4" not in workflow
    assert "scripts/select_state_space_parameters.py" in workflow
    assert "--min-momentum-recovery-accuracy" in workflow
    assert "--min-overall-recovery-accuracy" in workflow
    assert "state_space_parameter_decision_table.csv" in workflow
    assert "state_space_parameter_candidates.csv" in workflow
    assert "state_space_parameter_recommendation.csv" in workflow
    assert "state_space_parameter_selection_manifest.json" in workflow
    assert "state_space_selected_workflow_inputs.yml" in workflow
    assert "state_space_selected_cli_args.txt" in workflow
