"""Clock exclusions are fixed, preserve native identity and cannot admit data."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import audit_tanni_clock_interval_exclusion as audit
from scripts import verify_tanni_clock_interval_exclusion as verify
from tests.test_tanni_clock_source_reconciliation import make_source, source_nearest, source_convert


P = {"pulse_guard_count": 1, "acquisition_guard_s": .1, "maximum_camera_sample_distance_s": .1}
PARENT = {"global_clock_rising_channel": 1, "camera_clock_divider": 1e6,
          "multicamera_processed_sampling_rate_hz": 30, "clock_tolerance_s": .005, "coordinate_tolerance_cm": .05}


def test_original_frame_lookup_including_duplicates_and_ties():
    times = np.asarray([9., 1., 4., 4., 2., 7.])
    queries = np.asarray([-1., 1., 3., 4., 5.5, 8., 20.])
    expected = np.asarray([np.argmin(np.abs(times - q)) for q in queries])
    original = times.copy()
    np.testing.assert_array_equal(audit.native_nearest(times, queries), expected)
    np.testing.assert_array_equal(verify.reference_frame_nearest(times, queries), expected)
    np.testing.assert_array_equal(times, original)


def test_random_raw_order_lookup_agrees_with_full_argmin():
    rng = np.random.default_rng(20261002)
    for _ in range(10):
        times = rng.integers(0, 30, 70).astype(float)
        queries = np.r_[rng.uniform(-2, 33, 40), np.arange(30) / 2]
        expected = [np.argmin(np.abs(times - q)) for q in queries]
        np.testing.assert_array_equal(audit.native_nearest(times, queries), expected)
        np.testing.assert_array_equal(verify.reference_frame_nearest(times, queries), expected)


def test_fixed_guard_preserves_original_pulse_identity():
    oe = np.arange(8.)
    camera = np.asarray([0., 1., 2., 1.9, 4., 5., 6., 7.])
    nearest = np.arange(8)
    converted = np.arange(8.) + .2
    original = camera.copy()
    intervals, bad, rows = audit.pulse_exclusions(oe, camera, converted, nearest, P)
    np.testing.assert_array_equal(camera, original)
    np.testing.assert_array_equal(bad, [False, True, True, True, True, False, False, False])
    np.testing.assert_allclose(intervals, [[.9, 4.3]])
    assert rows[0]["backward_pulse_index"] == 2
    other, other_bad, other_rows = verify.reference_exclusions(oe, camera, converted, nearest, P)
    np.testing.assert_array_equal(other, intervals)
    np.testing.assert_array_equal(other_bad, bad)
    assert other_rows == rows


def test_fixed_guard_includes_implicated_frames_outside_native_pulse_bounds():
    oe, camera = np.arange(5.), np.asarray([0., 1., .9, 3., 4.])
    intervals, _, _ = audit.pulse_exclusions(oe, camera, np.asarray([-.3, 3.4]), np.asarray([0, 3]), P)
    np.testing.assert_allclose(intervals, [[-.4, 3.5]])


def test_unchanged_camera_has_no_clock_exclusions():
    intervals, bad, rows = audit.pulse_exclusions(np.arange(4.), np.arange(4.), np.arange(4.), np.arange(4), P)
    assert intervals.shape == (0, 2) and not bad.any() and rows == []


def test_exclusion_runs_keep_original_row_indices_and_gap_boundaries():
    rows = audit.mask_runs([False, True, True, False, True], np.arange(5.))
    assert [(r["first_processed_row"], r["last_processed_row"]) for r in rows] == [(1, 2), (4, 4)]
    assert sum(r["excluded_rows"] for r in rows) == 3


def test_geometry_resets_at_exclusions_and_grid_anchor_does_not_shift():
    times = np.arange(8.)
    camera = np.column_stack((times, np.ones((8, 4))))
    calls = []
    def combine(positions, previous, *args):
        calls.append(previous)
        return np.ones(4) if previous is None else previous + 1
    full, safe, good = audit.rebuild({"1": camera, "2": camera}, {}, [10, 10], combine,
        lambda a, *args, **kwargs: a, 1., np.asarray([[2., 3.]]),
        {"1": np.ones(8, bool), "2": np.ones(8, bool)}, .1)
    np.testing.assert_array_equal(safe[:, 0], np.arange(7.))
    assert not good[2:4].any() and np.isnan(safe[2:4, 1:]).all()
    assert safe[4, 1] == 1 and full[4, 1] == 5
    matched, *_ = audit.row_match(full, safe, audit.native_nearest, .005, .05)
    assert not matched[4]


def test_distant_camera_sample_is_an_explicit_gap_not_interpolation():
    t = np.arange(4.)
    first = np.column_stack((t, np.ones((4, 4))))
    second = first[[0, 3]]
    _, safe, good = audit.rebuild({"1": first, "2": second}, {}, [10, 10],
        lambda positions, *args: positions[0], lambda a, *args, **kwargs: a,
        1., [], {"1": np.ones(4, bool), "2": np.ones(2, bool)}, .1)
    np.testing.assert_array_equal(good, [True, False, False])
    assert np.isnan(safe[1:, 1:]).all()


def test_single_camera_unchanged_source_remains_diagnostic_only(tmp_path):
    path = tmp_path / "source.nwb"
    make_source(path)
    result, _, defects, masks, _ = audit.inspect_file(path, PARENT, P, None,
        lambda a, *args, **kwargs: a, source_nearest, source_convert)
    assert result["tracking_exclusion_reconciled"] and result["retained_rows"] == 4
    assert result["excluded_rows"] == 0 and masks == defects == []
    assert result["eligible_for_neural_analysis"] is False


def test_coordinate_mismatch_cannot_be_dropped_to_get_a_pass(tmp_path):
    path = tmp_path / "source.nwb"
    make_source(path)
    with h5py.File(path, "r+") as h:
        h["acquisition/timeseries/recording/tracking/ProcessedPos"][2, 1] += 1
    result, _, _, masks, _ = audit.inspect_file(path, PARENT, P, None,
        lambda a, *args, **kwargs: a, source_nearest, source_convert)
    assert not result["tracking_exclusion_reconciled"]
    assert result["retained_rows"] == 4 and result["retained_mismatched_rows"] == 1
    assert masks == []


@pytest.mark.parametrize("field,value", [
    ("OnlineTrackerData_timestamps", [1., 0., 2., 3.]),
    ("OnlineTrackerData_timestamps", [0., 0., 2., 3.]),
    ("GlobalClock_timestamps", [0., 0.]),
    ("GlobalClock_timestamps", [0., np.nan]),
])
def test_invalid_raw_clocks_are_not_fixed_by_exclusion(tmp_path, field, value):
    path = tmp_path / "source.nwb"
    make_source(path)
    with h5py.File(path, "r+") as h:
        h[f"acquisition/timeseries/recording/tracking/1/{field}"][...] = value
    with pytest.raises(ValueError):
        audit.inspect_file(path, PARENT, P, None, lambda a, *args, **kwargs: a,
                           source_nearest, source_convert)


def test_protocol_has_no_neural_or_association_admission():
    p = json.loads((Path(audit.__file__).parents[1] / "docs/tanni_clock_interval_exclusion_protocol.json").read_text())
    assert p["pulse_guard_count"] == 1 and not p["neural_analysis_enabled"] and not p["cohort_changed"]
    assert "never enlarge" in p["invalid_interval_rule"]
    assert "Any mismatch rejects the whole recording" in p["tracking_rule"]
