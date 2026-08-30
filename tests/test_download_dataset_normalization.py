from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "actions"
    / "download-pfeiffer-foster"
    / "normalize_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("download_pfeiffer_foster_normalize_dataset", _SCRIPT)
assert _SPEC is not None
normalize_dataset = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(normalize_dataset)


def _write_valid_session(session: Path) -> None:
    session.mkdir(parents=True)
    for name in normalize_dataset.REQUIRED_SESSION_FILES:
        (session / name).write_bytes(b"test")


def test_find_dataset_roots_discovers_nested_webdav_wrapper(tmp_path: Path) -> None:
    staging = tmp_path / "pfeiffer-foster-webdav"
    dataset = staging / "public-share" / "download" / "DataSetFromPfeifferFoster"
    _write_valid_session(dataset / "Rat1" / "Open1")

    assert normalize_dataset.find_dataset_roots(staging) == [dataset]


def test_find_dataset_roots_deduplicates_multiple_sessions(tmp_path: Path) -> None:
    staging = tmp_path / "pfeiffer-foster-webdav"
    dataset = staging / "wrapped" / "DataSetFromPfeifferFoster"
    _write_valid_session(dataset / "Rat1" / "Open1")
    _write_valid_session(dataset / "Rat2" / "Open2")

    assert normalize_dataset.find_dataset_roots(staging) == [dataset]


def test_main_rejects_multiple_nested_dataset_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner_temp = tmp_path / "runner"
    staging = runner_temp / "pfeiffer-foster-webdav"
    first = staging / "download-a" / "DataSetFromPfeifferFoster"
    second = staging / "download-b" / "DataSetFromPfeifferFoster"
    _write_valid_session(first / "Rat1" / "Open1")
    _write_valid_session(second / "Rat2" / "Open2")
    dataset_root = tmp_path / "prepared" / "DataSetFromPfeifferFoster"

    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    monkeypatch.setenv("DATASET_ROOT", str(dataset_root))

    with pytest.raises(SystemExit, match="multiple Pfeiffer-Foster dataset roots"):
        normalize_dataset.main()

    assert not dataset_root.exists()
    assert first.is_dir()
    assert second.is_dir()
