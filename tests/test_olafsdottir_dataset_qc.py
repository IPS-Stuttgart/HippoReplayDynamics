from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "summarize_olafsdottir_dataset_qc.py"
    spec = importlib.util.spec_from_file_location("summarize_olafsdottir_dataset_qc", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_track_sleep_pair_qc_marks_usable_and_excluded_pairs() -> None:
    module = _load_module()
    pairs = module.build_track_sleep_pairs(_synthetic_manifest())

    usable = pairs[pairs["animal"].eq("R2142")].iloc[0]
    assert bool(usable["usable_pair"])
    assert bool(usable["r2142_reversal_applied"])
    assert bool(usable["track_has_pos"])
    assert bool(usable["sleep_has_egf"])

    missing_pos = pairs[pairs["animal"].eq("R2335")].iloc[0]
    assert not bool(missing_pos["usable_pair"])
    assert "track_missing_pos" in missing_pos["exclusion_reason"]

    missing_sleep = pairs[pairs["animal"].eq("R2336")].iloc[0]
    assert not bool(missing_sleep["usable_pair"])
    assert "no_sleepPOST" in missing_sleep["exclusion_reason"]

    ambiguous = pairs[pairs["animal"].eq("R2337")].iloc[0]
    assert not bool(ambiguous["usable_pair"])
    assert "multiple_track1" in ambiguous["exclusion_reason"]
    assert ";" in ambiguous["track_session"]


def test_qc_summary_reports_counts_and_recommendation(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _synthetic_manifest()
    manifest_path = tmp_path / "olafsdottir2016_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    paths = module.write_qc_outputs(manifest_path, tmp_path / "out")

    pairs = pd.read_csv(paths["pairs"])
    summary = paths["summary"].read_text(encoding="utf-8")
    assert list(pairs.columns) == module.PAIR_COLUMNS
    assert pairs["usable_pair"].sum() == 1
    assert "Animals | 4" in summary
    assert "Usable pairs | 1" in summary
    assert "R2142 reversal check | pass" in summary
    assert "feasibility smoke" in summary
    assert "track_missing_pos" in summary


def test_manifest_loader_rejects_missing_columns(tmp_path: Path) -> None:
    module = _load_module()
    bad_manifest = tmp_path / "bad.csv"
    pd.DataFrame([{"animal": "R2142"}]).to_csv(bad_manifest, index=False)

    try:
        module.load_manifest(bad_manifest)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "session_type" in str(exc)
    else:
        raise AssertionError("load_manifest should reject incomplete manifests")


def _synthetic_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("R2142", "2014-08-06", "track1", "20140806_R2142_track1", has_pos=True, n_cut_files=2, n_egf_files=2, hpc="1,2,3,4,5,6,7,8", mec="9,10,11,12,13,14,15,16"),
            _row("R2142", "2014-08-06", "sleepPOST", "20140806_R2142_sleepPOST", has_pos=False, n_cut_files=2, n_egf_files=2, hpc="1,2,3,4,5,6,7,8", mec="9,10,11,12,13,14,15,16"),
            _row("R2335", "2015-10-26", "track1", "20151026_R2335_track1", has_pos=False, n_cut_files=1, n_egf_files=1),
            _row("R2335", "2015-10-26", "sleepPOST", "20151026_R2335_sleepPOST", has_pos=False, n_cut_files=1, n_egf_files=1),
            _row("R2336", "2015-11-01", "track1", "20151101_R2336_track1", has_pos=True, n_cut_files=1, n_egf_files=1),
            _row("R2337", "2015-11-03", "track1", "20151103_R2337_track1_a", has_pos=True, n_cut_files=1, n_egf_files=1),
            _row("R2337", "2015-11-03", "track1", "20151103_R2337_track1_b", has_pos=True, n_cut_files=1, n_egf_files=1),
            _row("R2337", "2015-11-03", "sleepPOST", "20151103_R2337_sleepPOST", has_pos=False, n_cut_files=1, n_egf_files=1),
        ]
    )


def _row(
    animal: str,
    date: str,
    session_type: str,
    session_name: str,
    *,
    has_pos: bool,
    n_cut_files: int,
    n_egf_files: int,
    hpc: str = "9,10,11,12,13,14,15,16",
    mec: str = "1,2,3,4,5,6,7,8",
) -> dict[str, object]:
    return {
        "animal": animal,
        "date": date,
        "session_type": session_type,
        "session_name": session_name,
        "session_path": f"/data/{animal}/{date}/{session_name}",
        "has_pos": str(has_pos).lower(),
        "has_set": "true",
        "n_cut_files": n_cut_files,
        "n_egf_files": n_egf_files,
        "n_tetrode_files": max(n_cut_files, n_egf_files),
        "hippocampal_tetrodes": hpc,
        "mec_tetrodes": mec,
        "notes": "",
    }
