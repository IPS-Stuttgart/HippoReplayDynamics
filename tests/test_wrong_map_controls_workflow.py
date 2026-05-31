from pathlib import Path


def test_wrong_map_controls_workflow_runs_sharded_calibrated_controls():
    workflow = Path(".github/workflows/wrong-map-evidence-controls.yml").read_text(
        encoding="utf-8"
    )
    script = Path("scripts/wrong_map_evidence_controls.py").read_text(encoding="utf-8")

    assert "name: Wrong-map replay evidence controls" in workflow
    assert "session_map_pairs:" in workflow
    assert "Rat1/Open1=Rat1/Open2" in workflow
    assert "scripts/plan_model_evidence_event_shards.py" in workflow
    assert "scripts/wrong_map_evidence_controls.py" in workflow
    assert "scripts/compare_wrong_map_evidence_controls.py" in workflow
    assert "--map-session" in workflow
    assert "real_evidence_run_id:" in workflow
    assert "real_evidence_artifact_name:" in workflow
    assert "Download real-map reference artifact" in workflow
    assert "--state-space-valid-occupancy-threshold-s" in workflow
    assert "--state-space-momentum-initial-sigma-cm-sqrt-s" in workflow
    assert "--emission-likelihood-temperature" in workflow
    assert "--emission-negative-binomial-overdispersion" in workflow
    assert 'CLUSTERLESS_MARK_VARIANCE_FLOOR: "1.0"' in workflow
    assert 'CLUSTERLESS_RATE_FLOOR_HZ: "0.0001"' in workflow
    assert 'CLUSTERLESS_MARK_LIKELIHOOD: "local-kde"' in workflow
    assert "wrong_map_control_gate_summary.csv" in workflow
    assert "wrong-map-evidence-controls-${{ github.run_id }}" in workflow

    assert "emission_likelihood_temperature" in script
    assert "negative_binomial_overdispersion" in script
    assert "--state-space-valid-occupancy-threshold-s" in script
