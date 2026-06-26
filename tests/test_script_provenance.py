from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_provenance_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "_provenance.py"
    spec = importlib.util.spec_from_file_location("script_provenance", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_command_line_preserves_explicit_empty_argv() -> None:
    module = _load_provenance_module()

    assert module.command_line([]) == ""


def test_build_script_provenance_preserves_explicit_empty_argv(tmp_path: Path) -> None:
    module = _load_provenance_module()

    provenance = module.build_script_provenance(argv=[], cwd=tmp_path)

    assert provenance["command_line"] == ""
