from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_olafsdottir2016_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_olafsdottir2016_dataset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_force_download_cli_implies_download(monkeypatch, tmp_path: Path) -> None:
    module = _load_prepare_script()
    captured: dict[str, object] = {}

    def fake_prepare_dataset(**kwargs):
        captured.update(kwargs)
        return tmp_path / "manifest.csv", []

    monkeypatch.setattr(module, "prepare_dataset", fake_prepare_dataset)

    result = module.main(
        [
            "--dataset-root",
            str(tmp_path),
            "--archive-path",
            str(tmp_path / "Olafsdottir2016.zip"),
            "--expected-md5",
            "",
            "--force-download",
        ]
    )

    assert result == 0
    assert captured["download"] is True
    assert captured["force_download"] is True
