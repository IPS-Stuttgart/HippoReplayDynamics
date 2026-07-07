from __future__ import annotations

from pathlib import Path

import pytest

from hipporeplayimm.olafsdottir_zip_safety import _safe_zip_member_path


def test_safe_zip_member_path_rejects_nul_byte(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe zip member path"):
        _safe_zip_member_path(tmp_path.resolve(), "safe\x00evil.txt")


def test_safe_zip_member_path_allows_nested_dataset_member(tmp_path: Path) -> None:
    root = tmp_path.resolve()

    destination = _safe_zip_member_path(root, "Olafsdottir2016/R2142/track1.set")

    assert destination == root / "Olafsdottir2016" / "R2142" / "track1.set"


def test_safe_zip_member_path_resolves_relative_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    destination = _safe_zip_member_path(Path("dataset"), "Olafsdottir2016/R2142/track1.set")

    assert destination == (tmp_path / "dataset" / "Olafsdottir2016" / "R2142" / "track1.set").resolve()
