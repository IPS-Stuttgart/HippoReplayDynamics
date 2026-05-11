from pathlib import Path


def test_position_validation_matrix_workflow_uploads_ranked_summary():
    workflow = Path(".github/workflows/position-validation-matrix.yml").read_text(encoding="utf-8")

    assert "name: Position validation parameter matrix" in workflow
    assert "sessions:" in workflow
    assert "decode_bin_s_values:" in workflow
    assert "smoothing_sigma_bins_values:" in workflow
    assert "passes_smoke_gate" in workflow
    assert "position_validation_matrix_ranked.csv" in workflow
    assert "position_validation_matrix_best_by_session.csv" in workflow
    assert "pattern: position-validation-matrix-*" in workflow
