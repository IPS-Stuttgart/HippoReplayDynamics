from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_validate_dataset_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / ".github" / "actions" / "ensure-pfeiffer-foster-dataset" / "validate_dataset.py"
    spec = importlib.util.spec_from_file_location("validate_dataset_action", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_validate_dataset_writes_checksumed_json_manifest(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "DataSetFromPfeifferFoster"
    session = dataset_root / "Rat1" / "Open1"
    session.mkdir(parents=True)
    file_payloads = {
        "Position_Data.mat": b"position",
        "Ripple_Events.mat": b"ripples",
        "Spike_Data.mat": b"spikes",
        "Epochs.mat": b"epochs",
        "Well_Sequence.mat": b"wells",
    }
    for filename, payload in file_payloads.items():
        (session / filename).write_bytes(payload)
    (dataset_root / "MANIFEST.txt").write_text("legacy manifest\n", encoding="utf-8")
    (dataset_root / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")

    module = _load_validate_dataset_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_dataset.py",
            "--dataset-root",
            str(dataset_root),
            "--session",
            "Rat1/Open1",
            "--write-manifest",
        ],
    )

    module.main()

    manifest_path = dataset_root / "dataset_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_root_name"] == "DataSetFromPfeifferFoster"
    assert manifest["session_count"] == 1
    assert manifest["total_bytes"] == sum(len(payload) for payload in file_payloads.values())
    assert {record["path"] for record in manifest["files"]} == {
        f"Rat1/Open1/{filename}" for filename in file_payloads
    }
    assert "MANIFEST.txt" not in {record["path"] for record in manifest["files"]}
    assert "dataset_manifest.json" not in {record["path"] for record in manifest["files"]}

    position_record = next(
        record for record in manifest["files"] if record["path"] == "Rat1/Open1/Position_Data.mat"
    )
    assert position_record["size_bytes"] == len(file_payloads["Position_Data.mat"])
    assert position_record["sha256"] == _sha256(file_payloads["Position_Data.mat"])

    session_record = manifest["sessions"][0]
    assert session_record["session"] == "Rat1/Open1"
    assert session_record["missing_required_files"] == []
    assert {record["path"] for record in session_record["required_files"]} == {
        "Rat1/Open1/Position_Data.mat",
        "Rat1/Open1/Ripple_Events.mat",
        "Rat1/Open1/Spike_Data.mat",
        "Rat1/Open1/Epochs.mat",
    }
    assert [record["path"] for record in session_record["optional_files"]] == [
        "Rat1/Open1/Well_Sequence.mat"
    ]

    self_hash = manifest.pop("manifest_sha256_without_this_field")
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    assert self_hash == _sha256(payload.encode("utf-8"))
