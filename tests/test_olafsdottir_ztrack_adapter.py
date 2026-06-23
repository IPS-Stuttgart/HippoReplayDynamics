from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np
import scipy.io as sio


def _load_adapter_module():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    module_path = repo_root / "scripts" / "prepare_olafsdottir_ztrack_sessions.py"
    spec = importlib.util.spec_from_file_location("prepare_olafsdottir_ztrack_sessions", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_region_mapping_uses_reversed_r2142_assignment() -> None:
    module = _load_adapter_module()

    assert module.hippocampal_tetrodes("r2142", "hippocampus") == tuple(range(1, 9))
    assert module.hippocampal_tetrodes("r2335", "hippocampus") == tuple(range(9, 17))
    assert module.hippocampal_tetrodes("r2335", "all") == tuple(range(1, 17))


def test_geodesic_linearization_keeps_bent_track_order() -> None:
    module = _load_adapter_module()
    xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [20.0, 10.0],
            [20.0, 20.0],
        ]
    )

    linear = module.linearize_position_geodesic(xy, bin_size_cm=10.0)

    assert np.all(np.diff(linear) >= 0.0) or np.all(np.diff(linear) <= 0.0)
    assert np.nanmax(linear) - np.nanmin(linear) >= 30.0


def test_mean_envelope_detector_finds_synthetic_event() -> None:
    module = _load_adapter_module()
    traces = np.zeros((2, 1000), dtype=float)
    traces[:, 400:430] = 4.0
    traces[:, 380:450] += 1.0
    config = module.RippleDetectionConfig(
        channel_numbers=(1, 2),
        band_low_hz=150.0,
        band_high_hz=250.0,
        high_threshold_z=3.0,
        low_threshold_z=0.5,
        min_duration_s=0.01,
        max_duration_s=0.2,
        expand_to_s=0.0,
        exclude_sleep_onset_s=0.0,
        detector_mode="mean-envelope",
        consensus_min_channels=2,
    )

    events, event_table = module.detect_ripple_events_from_traces(traces, 1000.0, config)

    assert events.shape == (1, 6)
    assert 0.39 <= events[0, 2] <= 0.43
    assert int(event_table.loc[0, "channels_above_high_at_peak"]) == 2


def test_build_session_writes_bridge_mat_files_and_metadata(tmp_path: Path) -> None:
    module = _load_adapter_module()
    day_dir = tmp_path / "extracted" / "r2142" / "2014-08-06"
    day_dir.mkdir(parents=True)
    track_stem = "20140806_R2142_track1"
    sleep_stem = "20140806_R2142_sleepPOST"
    _write_set(day_dir / f"{track_stem}.set", duration_s=3.0)
    _write_set(day_dir / f"{sleep_stem}.set", duration_s=2.0)
    _write_pos(day_dir / f"{track_stem}.pos")
    _write_tetrode(day_dir / f"{track_stem}.1", [0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    _write_cut(day_dir / f"{track_stem}_1.cut", [1, 2, 3, 1, 2, 3])
    _write_tetrode(day_dir / f"{sleep_stem}.1", [0.505, 0.510, 0.515, 0.520, 0.525, 0.530])
    _write_cut(day_dir / f"{sleep_stem}_1.cut", [1, 2, 3, 1, 2, 3])
    _write_egf_with_ripple(day_dir / f"{sleep_stem}.egf")

    day = module.DayPair(
        animal="r2142",
        date="2014-08-06",
        day_dir=day_dir,
        track_stem=track_stem,
        sleep_stem=sleep_stem,
    )
    config = module.ConversionConfig(
        tetrode_mode="hippocampus",
        min_track_spikes=1,
        min_sleep_spikes=1,
        max_sleep_rate_hz=100.0,
        min_event_spikes=5,
        min_event_active_cells=3,
        linear_position_bin_cm=10.0,
        sleep_offset_padding_s=10.0,
        ripple=module.RippleDetectionConfig(
            channel_numbers=(1,),
            band_low_hz=150.0,
            band_high_hz=250.0,
            high_threshold_z=1.5,
            low_threshold_z=0.2,
            min_duration_s=0.01,
            max_duration_s=0.6,
            expand_to_s=0.2,
            exclude_sleep_onset_s=0.0,
            detector_mode="mean-envelope",
            consensus_min_channels=1,
        ),
    )

    summary = module.build_session(day, tmp_path / "derived", config)

    session_dir = tmp_path / "derived" / "R2142" / "ZTrack20140806"
    assert (session_dir / "Position_Data.mat").is_file()
    assert (session_dir / "Spike_Data.mat").is_file()
    assert (session_dir / "Ripple_Events.mat").is_file()
    assert (session_dir / "Epochs.mat").is_file()
    assert (session_dir / "Experiment_Information.mat").is_file()
    assert summary["source_dataset"] == "Olafsdottir2016"
    assert summary["source_animal"] == "R2142"
    assert summary["adapter_schema_version"] == "olafsdottir_ztrack_pfeiffer_bridge_v1"
    assert summary["ripple_events"] == 1
    assert summary["sleep_time_offset_s"] > summary["track_duration_s"]

    info = sio.loadmat(session_dir / "Experiment_Information.mat", squeeze_me=True)
    assert str(info["source_dataset"]) == "Olafsdottir2016"
    assert str(info["source_track_session"]) == track_stem
    assert float(info["sleep_time_offset_s"]) == summary["sleep_time_offset_s"]
    filter_params = json.loads(str(info["event_filter_parameters"]))
    assert filter_params["min_event_spikes"] == 5
    assert filter_params["min_event_active_cells"] == 3

    ripples = sio.loadmat(session_dir / "Ripple_Events.mat", squeeze_me=True)["Ripple_Events"]
    assert np.asarray(ripples).reshape(-1, 6).shape == (1, 6)
    epochs = sio.loadmat(session_dir / "Epochs.mat", squeeze_me=True)
    run_times = np.asarray(epochs["Run_Times"], dtype=float).reshape(-1, 2)
    sleep_times = np.asarray(epochs["Sleep_Times"], dtype=float).reshape(-1, 2)
    assert run_times[0, 1] < sleep_times[0, 0]


def _write_set(path: Path, *, duration_s: float) -> None:
    path.write_text(f"duration {duration_s}\n", encoding="ascii")


def _write_pos(path: Path) -> None:
    header = (
        "timebase 50 hz\n"
        "sample_rate 50.0 hz\n"
        "pixels_per_metre 100\n"
        "num_pos_samples 6\n"
        "data_start"
    ).encode("ascii")
    records = b"".join(
        struct.pack(">I8h", t, x, 20, 1023, 1023, 1, 0, 0, 0)
        for t, x in [(0, 0), (50, 10), (100, 20), (150, 20), (200, 20), (250, 20)]
    )
    path.write_bytes(header + records)


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


def _write_egf_with_ripple(path: Path) -> None:
    sample_rate = 4800
    times = np.arange(sample_rate, dtype=float) / float(sample_rate)
    signal = np.zeros(sample_rate, dtype=float)
    burst = (times >= 0.49) & (times <= 0.55)
    signal[burst] = 3000.0 * np.sin(2.0 * np.pi * 200.0 * times[burst])
    payload = np.asarray(signal, dtype=">i2").tobytes()
    header = (
        "sample_rate 4800 hz\n"
        f"num_EGF_samples {sample_rate}\n"
        "data_start"
    ).encode("ascii")
    path.write_bytes(header + payload)
