from __future__ import annotations

import csv
from pathlib import Path
import zipfile

import pytest

from hipporeplayimm.olafsdottir2016 import (
    build_manifest,
    extract_archive,
    infer_session_type,
    prepare_dataset,
    tetrode_arrangement_for_animal,
    verify_md5,
    write_manifest_csv,
)


def _touch(path: Path, payload: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_tetrode_arrangement_handles_r2142_reversal() -> None:
    r2142 = tetrode_arrangement_for_animal("r2142")
    standard = tetrode_arrangement_for_animal("r2335")

    assert r2142.hippocampal_tetrodes == tuple(range(1, 9))
    assert r2142.mec_tetrodes == tuple(range(9, 17))
    assert "reversed" in r2142.notes
    assert standard.hippocampal_tetrodes == tuple(range(9, 17))
    assert standard.mec_tetrodes == tuple(range(1, 9))


def test_infer_session_type_from_axona_stem() -> None:
    assert infer_session_type("20140806_R2142_track1") == "track1"
    assert infer_session_type("20140806_R2142_sleepPOST") == "sleepPOST"
    assert infer_session_type("20140806_R2142_Training") == "Training"
    assert infer_session_type("20140806_R2142_screening") == "Screening"


def test_build_manifest_counts_synthetic_axona_files(tmp_path: Path) -> None:
    day = tmp_path / "r2142" / "2014-08-06"
    track = day / "20140806_R2142_track1"
    sleep = day / "20140806_R2142_sleepPOST"
    for stem in (track, sleep):
        _touch(stem.with_suffix(".set"))
        _touch(stem.with_suffix(".pos"))
        _touch(stem.with_suffix(".egf"))
        _touch(day / f"{stem.name}.egf2")
        _touch(day / f"{stem.name}.1")
        _touch(day / f"{stem.name}.2")
        _touch(day / f"{stem.name}_1.cut")
        _touch(day / f"{stem.name}_2.cut")
        _touch(day / f"{stem.name}.clu.1")

    records = build_manifest(tmp_path)

    assert [record.session_type for record in records] == ["sleepPOST", "track1"]
    track_record = next(record for record in records if record.session_type == "track1")
    assert track_record.animal == "R2142"
    assert track_record.date == "2014-08-06"
    assert track_record.has_pos
    assert track_record.has_set
    assert track_record.n_cut_files == 2
    assert track_record.n_egf_files == 2
    assert track_record.n_tetrode_files == 2
    assert track_record.hippocampal_tetrodes == tuple(range(1, 9))
    assert track_record.mec_tetrodes == tuple(range(9, 17))
    assert "reversed" in track_record.notes


def test_manifest_csv_writes_expected_columns(tmp_path: Path) -> None:
    day = tmp_path / "r2335" / "2015-10-26"
    stem = day / "20151026_R2335_Training"
    _touch(stem.with_suffix(".set"))
    _touch(stem.with_suffix(".pos"))
    _touch(stem.with_suffix(".egf"))
    _touch(day / f"{stem.name}.9")
    _touch(day / f"{stem.name}_9.cut")

    manifest_path = write_manifest_csv(build_manifest(tmp_path), tmp_path / "manifest.csv")
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))

    assert len(rows) == 1
    assert rows[0]["animal"] == "R2335"
    assert rows[0]["session_type"] == "Training"
    assert rows[0]["hippocampal_tetrodes"] == "9,10,11,12,13,14,15,16"
    assert rows[0]["mec_tetrodes"] == "1,2,3,4,5,6,7,8"


def test_prepare_dataset_manifest_only_does_not_download(tmp_path: Path) -> None:
    day = tmp_path / "r2336" / "2015-11-01"
    stem = day / "20151101_R2336_track1"
    _touch(stem.with_suffix(".set"))

    manifest_path, records = prepare_dataset(
        dataset_root=tmp_path,
        zenodo_url="https://example.invalid/should-not-be-used.zip",
        expected_md5="",
        download=False,
        extract=False,
    )

    assert manifest_path == tmp_path / "olafsdottir2016_manifest.csv"
    assert manifest_path.exists()
    assert len(records) == 1
    assert not (tmp_path / "Olafsdottir2016.zip").exists()


def test_extract_archive_allows_nested_dataset_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Olafsdottir2016/R2142/2014-08-06/20140806_R2142_track1.set", "header")

    extract_archive(archive_path, tmp_path / "dataset")

    assert (tmp_path / "dataset" / "Olafsdottir2016" / "R2142" / "2014-08-06" / "20140806_R2142_track1.set").read_text() == "header"


@pytest.mark.parametrize("member_name", ["../escape.txt", r"..\escape.txt", "/tmp/escape.txt", "C:/escape.txt"])
def test_extract_archive_rejects_unsafe_member_paths(tmp_path: Path, member_name: str) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe/member.txt", "safe")
        archive.writestr(member_name, "bad")

    with pytest.raises(ValueError, match="Unsafe zip member path"):
        extract_archive(archive_path, tmp_path / "dataset")

    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "dataset" / "safe" / "member.txt").exists()


def test_verify_md5_rejects_mismatch(tmp_path: Path) -> None:
    payload = tmp_path / "mini.zip"
    payload.write_bytes(b"not really a zip")

    try:
        verify_md5(payload, "00000000000000000000000000000000")
    except ValueError as exc:
        assert "MD5 mismatch" in str(exc)
    else:
        raise AssertionError("verify_md5 should reject mismatched payloads")
