import pytest

from scripts.audit_sweep_completeness import audit_sweep_completeness


def test_sweep_completeness_rejects_empty_artifact_root(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="No planned matrix cells"):
        audit_sweep_completeness(
            artifact_root=root,
            output=tmp_path / "completeness.csv",
            mode="state-space-evidence",
        )

    assert not (tmp_path / "completeness.csv").exists()
    assert not (tmp_path / "completeness.summary.json").exists()
