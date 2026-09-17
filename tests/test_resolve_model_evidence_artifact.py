import pytest

from scripts.resolve_model_evidence_artifact import DEFAULT_PREFIX, _run_id_from_artifact, select_artifact


def test_select_artifact_uses_newest_unexpired_matching_artifact():
    artifacts = [
        {
            "id": 1,
            "name": f"{DEFAULT_PREFIX}100",
            "expired": False,
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "id": 2,
            "name": f"{DEFAULT_PREFIX}200",
            "expired": True,
            "created_at": "2026-09-01T00:00:00Z",
        },
        {
            "id": 3,
            "name": f"{DEFAULT_PREFIX}300",
            "expired": False,
            "created_at": "2026-08-01T00:00:00Z",
        },
        {
            "id": 4,
            "name": "unrelated-artifact",
            "expired": False,
            "created_at": "2026-09-10T00:00:00Z",
        },
    ]

    selected = select_artifact(artifacts)

    assert selected["name"] == f"{DEFAULT_PREFIX}300"


def test_select_artifact_honors_explicit_name_but_rejects_expired_artifact():
    artifacts = [
        {
            "id": 1,
            "name": f"{DEFAULT_PREFIX}100",
            "expired": True,
            "created_at": "2026-01-01T00:00:00Z",
        }
    ]

    with pytest.raises(RuntimeError, match="no unexpired"):
        select_artifact(artifacts, artifact_name=f"{DEFAULT_PREFIX}100")


def test_run_id_can_be_recovered_from_standard_artifact_name():
    artifact = {"name": f"{DEFAULT_PREFIX}27011374643"}

    assert _run_id_from_artifact(artifact, DEFAULT_PREFIX) == "27011374643"


def test_run_id_prefers_workflow_metadata_when_available():
    artifact = {
        "name": "custom-model-evidence-artifact",
        "workflow_run": {"id": 12345},
    }

    assert _run_id_from_artifact(artifact, DEFAULT_PREFIX) == "12345"
