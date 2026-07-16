from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import detect_hc11_lfp_ripples as detector  # noqa: E402


def test_phase_reversed_channels_do_not_cancel_power() -> None:
    sample_rate = 1250.0
    time = np.arange(int(sample_rate), dtype=float) / sample_rate
    envelope = ((time >= 0.40) & (time < 0.48)).astype(float)
    ripple = envelope * np.sin(2.0 * np.pi * 160.0 * time)
    channels = np.column_stack([ripple, -ripple])
    raw_average = channels.mean(axis=1)
    assert np.max(np.abs(raw_average)) < 1e-12
    power = detector.ripple_power_nss(channels, sample_rate)
    assert float(power[(time >= 0.41) & (time < 0.47)].mean()) > 100.0 * float(
        power[(time < 0.30)].mean() + 1e-12
    )


def test_ca1_shank_channel_fallback_uses_units_and_excludes_bad_channels() -> None:
    channels = detector.select_ca1_shank_channels(
        [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), np.array([8, 9])],
        np.array(["lCA1"] * 8 + [""] * 2, dtype=object),
        np.array([2]),
        np.array([1, 1, 1]),
        np.array([1, 2, 2]),
        np.array(["lCA1", "lCA1", "lCA1"], dtype=object),
    )
    np.testing.assert_array_equal(channels, [1, 5])


def test_threshold_detector_merges_and_applies_peak_and_duration_gates() -> None:
    sample_rate = 1000.0
    values = np.zeros(1000, dtype=float)
    values[100:125] = 3.0
    values[130:160] = 3.0
    values[140] = 7.0
    values[300:310] = 8.0  # Too short.
    values[500:540] = 3.0  # No high-threshold peak.
    events = detector.detect_threshold_events(
        values,
        [(0, len(values))],
        sample_rate,
        min_inter_event_ms=20.0,
        min_duration_ms=20.0,
        max_duration_ms=300.0,
    )
    assert len(events) == 1
    row = events.iloc[0]
    assert row["start_time_s"] == 0.1
    assert row["end_time_s"] == 0.16
    assert row["peak_time_s"] == 0.14


def test_native_peak_comparison_is_one_to_one() -> None:
    result = detector.compare_peak_times(
        np.array([1.00, 1.02, 2.00]),
        np.array([1.01, 2.01]),
        tolerance_s=0.025,
    )
    assert result["matched_events"] == 2
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0


def test_native_interval_comparison_is_one_to_one_and_iou_gated() -> None:
    detected = np.array([[1.00, 1.10], [1.05, 1.15], [2.00, 2.10]])
    native = np.array([[1.02, 1.12], [2.02, 2.12]])
    overlap = detector.compare_event_intervals(detected, native)
    assert overlap["matched_events"] == 2
    assert overlap["precision"] == 2 / 3
    assert overlap["recall"] == 1.0
    assert overlap["median_iou"] > 0.6

    strict = detector.compare_event_intervals(detected, native, min_iou=0.9)
    assert strict["matched_events"] == 0
    assert strict["precision"] == 0.0
    assert strict["recall"] == 0.0


def test_native_restrict_reader_handles_matlab_hdf_orientation(tmp_path: Path) -> None:
    path = tmp_path / "ripples.event.mat"
    with h5py.File(path, "w") as handle:
        params = handle.create_group("ripplesNREM/detectorparms")
        params.create_dataset("restrict", data=np.array([[1.0, 4.0], [2.0, 6.0]]))
    intervals = detector.load_native_restrict_intervals(path)
    np.testing.assert_array_equal(intervals, [[1.0, 2.0], [4.0, 6.0]])
