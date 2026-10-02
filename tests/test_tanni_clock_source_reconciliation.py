"""Published clock reproduction is a diagnostic, never chronology repair."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import audit_tanni_clock_source_reconciliation as audit
from scripts import verify_tanni_clock_source_reconciliation as verifier


def source_nearest(frames, clock):
    indices = clock.argsort()
    values = clock[indices]
    right = np.minimum(np.searchsorted(values, frames), len(values) - 1)
    use_left = (right > 0) & (np.abs(frames - values[np.maximum(right - 1, 0)]) < np.abs(frames - values[right]))
    return indices[right - use_left]


def source_convert(oe, camera, frames, other_times_divider):
    index = source_nearest(frames, camera)
    return oe[index] + (frames - camera[index]) / other_times_divider


def test_raw_pulse_defects_retained_not_sorted_or_offset_fitted():
    oe = np.asarray([10., 11., 12., 13.])
    camera = np.asarray([0., 1e6, .95e6, 3e6])
    frames = np.asarray([.1e6, .96e6, 1.02e6, 2.9e6])
    original = camera.copy()
    converted, stats, indices = audit.source_conversion(oe, camera, frames, 1e6, source_nearest, source_convert)
    np.testing.assert_array_equal(camera, original)
    np.testing.assert_array_equal(indices, [0, 2, 1, 3])
    np.testing.assert_allclose(converted, [10.1, 12.01, 11.02, 12.9])
    assert stats["raw_backward_steps"] == stats["converted_backward_steps"] == 1
    assert stats["frames_nearest_defective_pulse"] == 2


def test_exact_tie_uses_published_upper_clock_value():
    _, stats, indices = audit.source_conversion(np.asarray([10., 11.]), np.asarray([0., 1e6]),
                                               np.asarray([.5e6]), 1e6, source_nearest, source_convert)
    np.testing.assert_array_equal(indices, [1])
    assert stats["raw_backward_steps"] == 0


@pytest.mark.parametrize("oe,camera,frames,divider", [
    ([0., 1.], [0., 1., 2.], [.2], 1),
    ([0., 1., 2.], [0., 1., 1.], [.2], 1),
    ([0., 0.], [0., 1.], [.2], 1),
    ([0., 1.], [0., np.nan], [.2], 1),
    ([0., 1.], [0., 1.], [.4, .2], 1),
    ([0., 1.], [0., 1.], [.2], 0),
])
def test_no_truncation_duplicates_nonfinite_or_frame_clock_repair(oe, camera, frames, divider):
    with pytest.raises(ValueError):
        audit.source_conversion(np.asarray(oe), np.asarray(camera), np.asarray(frames), divider,
                                source_nearest, source_convert)


def test_pinned_converter_executes_only_two_functions(tmp_path, monkeypatch):
    folder = tmp_path / "openEPhys_DACQ"
    folder.mkdir()
    (folder / "HelperFunctions.py").write_text("raise RuntimeError('no module execution')\n"
        "def closest_argmin(a, b):\n    return np.zeros(len(a), dtype=int)\n")
    (folder / "NWBio.py").write_text("raise RuntimeError('no package I/O')\n"
        "def estimate_open_ephys_timestamps_from_other_timestamps(a, b, c, other_times_divider=None):\n"
        "    return a[closest_argmin(c, b)] + (c-b[0])/other_times_divider\n")
    monkeypatch.setattr(audit.subprocess, "check_output", lambda args, **kwargs: "abc\n" if "rev-parse" in args else "")
    nearest, converter, hashes = audit.pinned_converter(tmp_path, "abc")
    assert len(hashes) == 2
    np.testing.assert_array_equal(nearest(np.asarray([1]), np.asarray([0, 2])), [0])
    np.testing.assert_allclose(converter(np.asarray([10, 12]), np.asarray([0, 2]), np.asarray([1]), 1), [11])
    with pytest.raises(ValueError, match="commit differs"):
        audit.pinned_converter(tmp_path, "other")
    monkeypatch.setattr(audit.subprocess, "check_output", lambda args, **kwargs: "abc\n" if "rev-parse" in args else "dirty")
    with pytest.raises(ValueError, match="modified"):
        audit.pinned_converter(tmp_path, "abc")


def make_source(path):
    with h5py.File(path, "w") as h:
        prefix = "acquisition/timeseries/recording"
        frames = np.asarray([0.1, 0.2, 0.3, 0.4])
        xy = np.column_stack((np.arange(4), np.arange(4), np.arange(4), np.arange(4)))
        h[f"{prefix}/tracking/ProcessedPos"] = np.column_stack((frames, xy))
        h[f"{prefix}/tracking/1/OnlineTrackerData"] = xy
        h[f"{prefix}/tracking/1/OnlineTrackerData_timestamps"] = frames * 1e6
        h[f"{prefix}/tracking/1/GlobalClock_timestamps"] = np.asarray([0., 1e6])
        h[f"{prefix}/events/ttl1/timestamps"] = np.asarray([0., 1.])
        h[f"{prefix}/events/ttl1/data"] = np.asarray([1, 1])
        h["general/data_collection/Settings/General/arena_size"] = np.asarray([10., 10.])


def test_complete_position_reproduction_still_cannot_admit_source(tmp_path):
    path = tmp_path / "source.nwb"
    make_source(path)
    p = {"global_clock_rising_channel": 1, "camera_clock_divider": 1e6,
         "multicamera_processed_sampling_rate_hz": 30, "clock_tolerance_s": .005, "coordinate_tolerance_cm": .05}
    result, clocks, arrays = audit.inspect_file(path, p, None, lambda a, *args, **kwargs: a,
                                               source_nearest, source_convert)
    assert result["source_clock_reconciliation_passed"] and result["processed_rows"] == 4
    assert result["status"] == "source_reproduced_diagnostic_only"
    assert result["eligible_for_neural_analysis"] is False
    assert len(clocks) == 1 and len(arrays) == 7
    assert all("spikes" not in row["hdf5_path"] and "continuous" not in row["hdf5_path"] for row in arrays)
    with h5py.File(path, "r+") as h:
        h["acquisition/timeseries/recording/tracking/ProcessedPos"][2, 1] += 1
    result, _, _ = audit.inspect_file(path, p, None, lambda a, *args, **kwargs: a, source_nearest, source_convert)
    assert not result["source_clock_reconciliation_passed"] and not result["eligible_for_neural_analysis"]


def test_no_analysis_stage_or_cohort_amendment():
    text = Path(audit.__file__).read_text()
    assert '"cohort_changed": False' in text and '"association_fit": False' in text
    assert '"source_repair_applied": False' in text
    assert "--analysis" not in text and "--offset" not in text


def test_independent_lookup_matches_direct_distance_not_sorted_producer():
    rng = np.random.default_rng(20261002)
    clock = rng.permutation(np.arange(73, dtype=float))
    frames = np.r_[rng.uniform(-2, 75, 200), np.arange(72) + .5]
    expected = []
    for frame in frames:
        distance = np.abs(clock - frame)
        tied = np.flatnonzero(distance == distance.min())
        expected.append(tied[np.argmax(clock[tied])])
    np.testing.assert_array_equal(verifier.reference_nearest(frames, clock), expected)
    np.testing.assert_array_equal(verifier.reference_nearest(frames, clock), source_nearest(frames, clock))
    np.testing.assert_allclose(verifier.reference_convert(np.arange(73.), clock, frames, 1e6),
                               source_convert(np.arange(73.), clock, frames, 1e6))


def test_independent_lookup_rejects_ambiguous_identity():
    for clock, frames in [([1, 1], [1]), ([1, np.nan], [1]), ([1, 2], [np.nan])]:
        with pytest.raises(ValueError):
            verifier.reference_nearest(np.asarray(frames), np.asarray(clock))
