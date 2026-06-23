from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pandas as pd


def _load_detector_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    module_path = repo_root / "scripts" / "detect_olafsdottir_sleep_replay_events.py"
    spec = importlib.util.spec_from_file_location("detect_olafsdottir_sleep_replay_events", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detect_sleeppost_events_writes_required_tables(tmp_path: Path) -> None:
    module = _load_detector_module()
    day = tmp_path / "r2142" / "2014-08-06"
    day.mkdir(parents=True)
    track_stem = day / "20140806_R2142_track1"
    sleep_stem = day / "20140806_R2142_sleepPOST"
    track_stem.with_suffix(".set").write_text("duration 10\n", encoding="ascii")
    _write_egf_with_ripple(sleep_stem.with_suffix(".egf"))
    spike_times = [0.405, 0.41, 0.415, 0.42, 0.425, 0.43]
    labels = [1, 2, 3, 1, 2, 3]
    _write_cut(sleep_stem.with_name(f"{sleep_stem.name}_1.cut"), labels)
    _write_tetrode(sleep_stem.with_suffix(".1"), spike_times)

    manifest = pd.DataFrame(
        [
            _manifest_row("track1", track_stem),
            _manifest_row("sleepPOST", sleep_stem),
        ]
    )
    output = tmp_path / "events"

    events, summary = module.write_detection_outputs(
        manifest,
        output,
        min_event_spikes=5,
        min_event_active_cells=3,
        min_duration_s=0.01,
        max_duration_s=0.6,
        ripple_z_threshold=1.5,
        merge_gap_s=0.01,
        envelope_smooth_s=0.004,
    )

    assert (output / "sleep_replay_events.csv").is_file()
    assert (output / "ripple_detection_summary.csv").is_file()
    assert list(events.columns) == module.EVENT_COLUMNS
    assert list(summary.columns) == module.SUMMARY_COLUMNS
    assert len(events) == 1
    event = events.iloc[0]
    assert event["event_index"] == 0
    assert 0.15 <= event["start_time_s"] <= 0.25
    assert 0.40 <= event["peak_time_s"] <= 0.46
    assert event["n_spikes"] == 6
    assert event["n_active_cells"] == 3
    assert event["event_detector"] == "ripple_band_lfp_threshold_v1"
    params = json.loads(event["detector_parameters"])
    assert params["min_event_spikes"] == 5
    assert params["min_event_active_cells"] == 3

    row = summary.iloc[0]
    assert row["n_lfp_channels"] == 1
    assert row["n_events"] == 1
    assert row["median_event_spikes"] == 6.0
    assert row["max_event_spikes"] == 6
    assert "candidate detector" in row["caveat"]


def _manifest_row(session_type: str, stem: Path) -> dict[str, object]:
    return {
        "animal": "R2142",
        "date": "2014-08-06",
        "session_type": session_type,
        "session_name": stem.name,
        "session_path": str(stem),
        "has_pos": False,
        "has_set": True,
        "n_cut_files": 1,
        "n_egf_files": 1,
        "n_tetrode_files": 1,
        "hippocampal_tetrodes": "1,2,3,4,5,6,7,8",
        "mec_tetrodes": "9,10,11,12,13,14,15,16",
        "notes": "R2142 reversed arrangement: hippocampus=1-8, MEC=9-16",
    }


def _write_egf_with_ripple(path: Path) -> None:
    sample_rate = 4800
    times = np.arange(sample_rate, dtype=float) / float(sample_rate)
    signal = np.zeros(sample_rate, dtype=float)
    burst = (times >= 0.39) & (times <= 0.45)
    signal[burst] = 3000.0 * np.sin(2.0 * np.pi * 200.0 * times[burst])
    payload = np.asarray(signal, dtype=">i2").tobytes()
    header = (
        "sample_rate 4800 hz\n"
        f"num_EGF_samples {sample_rate}\n"
        "data_start"
    ).encode("ascii")
    path.write_bytes(header + payload)


def _write_cut(path: Path, labels: list[int]) -> None:
    path.write_text(
        f"Exact_cut_for: {path.name} spikes: {len(labels)}\n"
        + " ".join(str(label) for label in labels)
        + "\n",
        encoding="ascii",
    )


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
