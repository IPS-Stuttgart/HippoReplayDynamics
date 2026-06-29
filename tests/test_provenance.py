from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "_provenance.py"
    spec = importlib.util.spec_from_file_location("_provenance", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_relative_input_hashes_are_resolved_against_declared_cwd(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    work_dir = tmp_path / "work"
    outside_dir = tmp_path / "outside"
    work_dir.mkdir()
    outside_dir.mkdir()
    (work_dir / "input.csv").write_bytes(b"event_id,score\n1,2\n")
    monkeypatch.chdir(outside_dir)

    provenance = module.build_script_provenance(
        input_paths={"quality": "input.csv"},
        argv=["script.py", "--quality", "input.csv"],
        cwd=work_dir,
    )

    expected_hash = hashlib.sha256(b"event_id,score\n1,2\n").hexdigest()
    assert provenance["input_file_paths"] == {"quality": "input.csv"}
    assert provenance["input_file_sha256"] == {"quality": expected_hash}
