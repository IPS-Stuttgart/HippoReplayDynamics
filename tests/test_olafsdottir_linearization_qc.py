from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    scripts_path = repo_root / "scripts"
    for path in (src_path, scripts_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = scripts_path / "summarize_olafsdottir_linearization_qc.py"
    spec = importlib.util.spec_from_file_location("summarize_olafsdottir_linearization_qc", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_linearization_qc_writes_tables_and_figures(tmp_path: Path) -> None:
    module = _load_module()
    dataset_root = tmp_path / "data"
    track_stem = "20140806_R2142_track1"
    _write_track_session(dataset_root / "r2142" / "2014-08-06", track_stem)
    pairs = pd.DataFrame(
        [
            {
                "animal": "R2142",
                "date": "2014-08-06",
                "track_session": track_stem,
                "sleepPOST_session": "20140806_R2142_sleepPOST",
                "hippocampal_tetrodes": "1,2,3,4,5,6,7,8",
                "usable_pair": True,
            }
        ]
    )
    pairs_csv = tmp_path / "pairs.csv"
    pairs.to_csv(pairs_csv, index=False)

    tables = module.run_linearization_qc(
        dataset_root=dataset_root,
        pairs_csv=pairs_csv,
        output_dir=tmp_path / "qc",
        min_occupancy_nonzero_fraction=0.10,
    )

    sessions = tables["sessions"]
    gates = tables["gates"].set_index("gate")
    figures = tables["figures"]
    assert (tmp_path / "qc" / module.SESSION_OUTPUT).is_file()
    assert (tmp_path / "qc" / module.ANIMAL_OUTPUT).is_file()
    assert (tmp_path / "qc" / module.GATE_OUTPUT).is_file()
    assert (tmp_path / "qc" / module.FIGURE_OUTPUT).is_file()
    assert (tmp_path / "qc" / module.SUMMARY_OUTPUT).is_file()
    assert list(sessions.columns) == module.SESSION_COLUMNS
    row = sessions.iloc[0]
    assert row["linearization_status"] == "pass"
    assert row["n_position_samples"] == 51
    assert row["n_spikes_track1"] == 6
    assert row["n_units_track1"] == 2
    assert row["track_spike_position_overlap_s"] > 0
    assert row["occupancy_nonzero_bins"] > 0
    assert row["p95_off_track_distance_cm"] >= 0
    assert bool(row["reversal_applied"])
    assert gates["passed"].map(bool).all()
    assert set(figures["figure_type"]) == {
        "position_2d_centerline",
        "linear_position_over_time",
        "linear_occupancy",
        "off_track_distance_histogram",
        "speed_histogram",
    }
    assert all(Path(path).is_file() for path in figures["figure_path"])
    summary = (tmp_path / "qc" / module.SUMMARY_OUTPUT).read_text(encoding="utf-8")
    assert "Track1 sessions passing QC | 1" in summary


def test_linearization_qc_marks_missing_position_failure(tmp_path: Path) -> None:
    module = _load_module()
    dataset_root = tmp_path / "data"
    (dataset_root / "r2335" / "2015-10-26").mkdir(parents=True)
    pairs = pd.DataFrame(
        [
            {
                "animal": "R2335",
                "date": "2015-10-26",
                "track_session": "20151026_R2335_track1",
                "sleepPOST_session": "20151026_R2335_sleepPOST",
                "hippocampal_tetrodes": "9",
                "usable_pair": True,
            }
        ]
    )
    pairs_csv = tmp_path / "pairs.csv"
    pairs.to_csv(pairs_csv, index=False)

    tables = module.run_linearization_qc(
        dataset_root=dataset_root,
        pairs_csv=pairs_csv,
        output_dir=tmp_path / "qc",
    )

    row = tables["sessions"].iloc[0]
    assert row["linearization_status"] == "fail"
    assert "FileNotFoundError" in row["exclusion_reason"]
    gates = tables["gates"].set_index("gate")
    assert not bool(gates.loc["track1_position_samples_present", "passed"])


def test_load_pairs_rejects_missing_required_columns(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "pairs.csv"
    pd.DataFrame([{"animal": "R2142"}]).to_csv(path, index=False)

    try:
        module.load_pairs(path)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "track_session" in str(exc)
    else:
        raise AssertionError("load_pairs should reject incomplete input")


def _write_track_session(day_dir: Path, stem: str) -> None:
    day_dir.mkdir(parents=True)
    _write_pos(day_dir / f"{stem}.pos")
    _write_tetrode(day_dir / f"{stem}.1", [0.02, 0.08, 0.14, 0.20, 0.26, 0.34])
    _write_cut(day_dir / f"{stem}_1.cut", [1, 2, 1, 2, 1, 2])


def _write_pos(path: Path) -> None:
    samples = []
    for index in range(26):
        samples.append((index, index * 4, 0))
    for index in range(1, 26):
        samples.append((25 + index, 100, index * 4))
    header = (
        "timebase 50 hz\n"
        "sample_rate 50.0 hz\n"
        "bytes_per_timestamp 4\n"
        "bytes_per_coord 2\n"
        "pixels_per_metre 100\n"
        f"num_pos_samples {len(samples)}\n"
        "pos_format t,x1,y1,x2,y2,numpix1,numpix2\n"
        "data_start"
    ).encode("ascii")
    payload = b"".join(
        struct.pack(">I8h", frame, x, y, 1023, 1023, 1, 0, 1, 0)
        for frame, x, y in samples
    )
    path.write_bytes(header + payload)


def _write_tetrode(path: Path, spike_times_s: list[float]) -> None:
    header = (
        f"num_spikes {len(spike_times_s)}\n"
        "timebase 96000 hz\n"
        "samples_per_spike 2\n"
        "data_start"
    ).encode("ascii")
    payload = b"".join(
        struct.pack(">I", int(round(time_s * 96000.0))) + b"\x00" * 8
        for time_s in spike_times_s
    )
    path.write_bytes(header + payload)


def _write_cut(path: Path, labels: list[int]) -> None:
    path.write_text(
        f"Exact_cut_for: {path.name} spikes: {len(labels)}\n"
        + " ".join(str(label) for label in labels)
        + "\n",
        encoding="ascii",
    )
