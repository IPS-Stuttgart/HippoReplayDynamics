from pathlib import Path


def test_selfhosted_replay_alignment_resolves_unexpired_evidence_artifact():
    workflow = Path(
        ".github/workflows/replay-behavior-alignment-selfhosted.yml"
    ).read_text(encoding="utf-8")

    assert "27011374643" not in workflow
    assert "actions/github-script@v7" in workflow
    assert "listArtifactsForRepo" in workflow
    assert "candidate.expired" in workflow
    assert 'candidate.name.startsWith("model-evidence-all-sessions-")' in workflow
    assert "steps.evidence.outputs.run_id" in workflow
    assert "steps.evidence.outputs.artifact_name" in workflow
