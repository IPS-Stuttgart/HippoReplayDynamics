from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys

import numpy as np
import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    scripts_path = repo_root / "scripts"
    for path in (src_path, scripts_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_path = scripts_path / "summarize_olafsdottir_sleeppost_event_detection_qc.py"
    spec = importlib.util.spec_from_file_location("summarize_olafsdottir_sleeppost_event_detection_qc", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sleeppost_event_detection_qc_writes_outputs_and_gates(tmp_path: Path) -> None:
    module = _load_module()
    dataset_root = tmp_path / "data"
    rows = [
        ("R2142", "2014-08-06", "20140806_R2142_track1", "20140806_R2142_sleepPOST", "1"),
        ("R2335", "2015-10-26", "20151026_R2335_track1", "20151026_R2335_sleepPOST", "9"),
    ]
    for animal, date, _track, sleep, tetrode in rows:
        _write_sleep_session(dataset_root / animal.lower() / date, sleep, int(tetrode))
    pairs = pd.DataFrame(
        [
            {
                "animal": animal,
                "date": date,
                "track_session": track,
                "sleepPOST_session": sleep,
                "hippocampal_tetrodes": tetrode,
                "usable_pair": True,
            }
            for animal, date, track, sleep, tetrode in rows
        ]
    )
    linearization = pd.DataFrame(
        [
            {"animal": animal, "date": date, "track_session": track, "linearization_status": "pass"}
            for animal, date, track, _sleep, _tetrode in rows
        ]
    )
    pairs_csv = tmp_path / "pairs.csv"
    lin_csv = tmp_path / "linearization.csv"
    pairs.to_csv(pairs_csv, index=False)
    linearization.to_csv(lin_csv, index=False)

    tables = module.run_event_detection_qc(
        dataset_root=dataset_root,
        pairs_csv=pairs_csv,
        linearization_qc=lin_csv,
        output_dir=tmp_path / "qc",
        mua_z_threshold=1.5,
        min_dataset_candidate_events=2,
        min_dataset_candidate_sessions=2,
        min_paper_candidate_animals=2,
        max_paper_candidate_animal_fraction=0.75,
        max_paper_candidate_session_fraction=0.75,
    )

    assert (tmp_path / "qc" / module.SESSION_OUTPUT).is_file()
    assert (tmp_path / "qc" / module.EVENT_OUTPUT).is_file()
    assert (tmp_path / "qc" / module.ANIMAL_OUTPUT).is_file()
    assert (tmp_path / "qc" / module.GATE_OUTPUT).is_file()
    assert (tmp_path / "qc" / module.SUMMARY_OUTPUT).is_file()
    sessions = tables["sessions"]
    events = tables["events"]
    gates = tables["gates"].set_index("gate")
    assert list(sessions.columns) == module.SESSION_COLUMNS
    assert list(events.columns) == module.EVENT_COLUMNS
    assert len(events) >= 2
    assert sessions["candidate_event_count"].min() >= 1
    assert sessions["immobile_event_count"].min() >= 1
    assert events["n_active_units"].min() >= 3
    assert events["candidate_tier"].isin({"weak", "moderate", "strong", "extreme"}).all()
    assert events["event_qc_status"].eq("pass").all()
    assert gates["passed"].map(bool).all()
    summary = (tmp_path / "qc" / module.SUMMARY_OUTPUT).read_text(encoding="utf-8")
    assert "Dataset-usable gates passed | 2/2" in summary
    assert "Paper-ready gates passed | 4/4" in summary


def test_sleeppost_event_detection_qc_excludes_artifact_rows_from_counts() -> None:
    module = _load_module()
    events = pd.DataFrame(
        [
            {
                "animal": "R2335",
                "date": "2015-10-26",
                "session": "20151026_R2335_sleepPOST",
                "event_id": 0,
                "start_time_s": 0.0,
                "end_time_s": 0.02,
                "duration_ms": 20.0,
                "n_spikes": 120,
                "n_active_units": 3,
                "mean_mua_rate_hz": 6000.0,
                "peak_mua_rate_hz": 6000.0,
                "mean_speed_cm_s": 0.0,
                "event_detection_score": 20.0,
                "candidate_tier": "extreme",
                "event_qc_status": "artifact",
                "event_qc_reason": "recording_start_artifact;implausible_spikes_per_active_unit",
            },
            {
                "animal": "R2335",
                "date": "2015-10-26",
                "session": "20151026_R2335_sleepPOST",
                "event_id": 1,
                "start_time_s": 1.0,
                "end_time_s": 1.04,
                "duration_ms": 40.0,
                "n_spikes": 12,
                "n_active_units": 4,
                "mean_mua_rate_hz": 300.0,
                "peak_mua_rate_hz": 400.0,
                "mean_speed_cm_s": 0.0,
                "event_detection_score": 4.0,
                "candidate_tier": "moderate",
                "event_qc_status": "pass",
                "event_qc_reason": "",
            },
        ],
        columns=module.EVENT_COLUMNS,
    )
    row = module.session_row(
        animal="R2335",
        date="2015-10-26",
        sleep_session="20151026_R2335_sleepPOST",
        track_session="20151026_R2335_track1",
        sleep_duration=2.0,
        spikes=module.SleepSpikes(
            spike_times_s=np.asarray([1.0, 1.01, 1.02, 1.03], dtype=float),
            unit_ids=np.asarray([1, 2, 3, 4], dtype=int),
            unit_count=4,
        ),
        events=events,
        status="pass",
        reasons=[],
        immobility_speed_threshold_cm_s=5.0,
    )
    assert row["raw_candidate_event_count"] == 2
    assert row["artifact_flagged_event_count"] == 1
    assert row["candidate_event_count"] == 1
    assert row["median_event_spikes"] == 12.0
    assert row["immobile_event_count"] == 1


def test_sleeppost_event_detection_qc_marks_missing_spikes_failure(tmp_path: Path) -> None:
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
    linearization = pd.DataFrame([{"animal": "R2335", "date": "2015-10-26", "track_session": "20151026_R2335_track1", "linearization_status": "pass"}])
    pairs_csv = tmp_path / "pairs.csv"
    lin_csv = tmp_path / "linearization.csv"
    pairs.to_csv(pairs_csv, index=False)
    linearization.to_csv(lin_csv, index=False)

    tables = module.run_event_detection_qc(
        dataset_root=dataset_root,
        pairs_csv=pairs_csv,
        linearization_qc=lin_csv,
        output_dir=tmp_path / "qc",
        min_dataset_candidate_events=1,
        min_dataset_candidate_sessions=1,
        min_paper_candidate_animals=1,
    )

    row = tables["sessions"].iloc[0]
    assert row["event_detection_status"] == "fail"
    assert "missing_sleep_spike_timestamps_or_units" in row["exclusion_reason"] or "FileNotFoundError" in row["exclusion_reason"]
    assert not bool(tables["gates"].set_index("gate").loc["sleeppost_spike_data_present", "passed"])


def test_sleeppost_event_detection_qc_rejects_missing_inputs(tmp_path: Path) -> None:
    module = _load_module()
    pairs = tmp_path / "pairs.csv"
    linearization = tmp_path / "linearization.csv"
    pd.DataFrame([{"animal": "R2142"}]).to_csv(pairs, index=False)
    pd.DataFrame([{"animal": "R2142"}]).to_csv(linearization, index=False)

    try:
        module.load_pairs(pairs)
    except ValueError as exc:
        assert "sleepPOST_session" in str(exc)
    else:
        raise AssertionError("load_pairs should reject incomplete pair tables")

    try:
        module.load_linearization_qc(linearization)
    except ValueError as exc:
        assert "linearization_status" in str(exc)
    else:
        raise AssertionError("load_linearization_qc should reject incomplete QC tables")


def _write_sleep_session(day_dir: Path, stem: str, tetrode: int) -> None:
    day_dir.mkdir(parents=True)
    (day_dir / f"{stem}.set").write_text("duration 2.0\n", encoding="ascii")
    _write_pos(day_dir / f"{stem}.pos")
    spike_times = [0.50, 0.505, 0.510, 0.515, 0.520, 0.525, 1.20, 1.205, 1.210, 1.215, 1.220, 1.225]
    labels = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    _write_tetrode(day_dir / f"{stem}.{tetrode}", spike_times)
    _write_cut(day_dir / f"{stem}_{tetrode}.cut", labels)


def _write_pos(path: Path) -> None:
    samples = [(index, 100, 100) for index in range(101)]
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
