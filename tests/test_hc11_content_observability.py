from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from scripts.audit_hc11_content_observability import (
    COHORT,
    acquisition_clock_limit,
    count_windows,
    extract_one,
    group_readouts,
    native_position,
    native_units,
    partition,
    running_intervals,
    summarize,
    windows_for,
)
from scripts._provenance import file_sha256
from scripts.report_hc11_content_observability import audit_session, histogram_recount


def test_uid_join_not_row_order_and_missing_class_excluded():
    spikes = dict(UID=[3, 1, 5, 9], region=["lCA1", "rCA1", "CA1", "CA1"], shankID=[1, 2, 1, 2])
    classes = dict(UID=[5, 3, 1], pE=[0, 1, 1], pI=[1, 0, 0])
    result = native_units(spikes, classes)
    assert result.unit_id.tolist() == [3, 1, 5, 9]
    assert result.native_included.tolist() == [True, True, False, False]
    assert result.exclusion_reason.iloc[-1] == "missing_native_class"
    classes["UID"] = [5, 3, 3]
    with pytest.raises(ValueError, match="duplicate"):
        native_units(spikes, classes)


def source_position():
    t = np.arange(10, 30, .02)
    return dict(timestamps=t, position=dict(x=(t-10)*.2, y=np.zeros(len(t))), units="m",
                Epochs=dict(PREEpoch=[0, 10], MazeEpoch=[10, 30], POSTEpoch=[30, 100]),
                behaviorinfo=dict(MazeType="synthetic_linear"))


def test_position_metres_once_and_native_clock():
    source = source_position()
    position, epochs = native_position(source)
    np.testing.assert_allclose(position[:, 1], source["position"]["x"]*100)
    np.testing.assert_array_equal(position[:, 0], source["timestamps"])
    assert epochs[1, 0] == 10
    source["units"] = "cm"
    with pytest.raises(ValueError, match="units"):
        native_position(source)
    source["units"] = "m"
    source["timestamps"][3] = source["timestamps"][2]
    with pytest.raises(ValueError, match="clock"):
        native_position(source)


def test_run_screen_first_half_gaps_invalid_positions_and_edges():
    position, epochs = native_position(source_position())
    position[100:104, 1] = np.nan
    intervals = running_intervals(position, epochs[1])
    assert len(intervals) > 100
    assert intervals[:, 1].max() <= 20
    assert intervals[:, 0].min() > 10
    assert not ((intervals[:, 0] < position[105, 0]) & (intervals[:, 1] > position[98, 0])).any()
    altered = position.copy()
    altered[altered[:, 0] >= 20, 1:] = np.nan
    np.testing.assert_array_equal(running_intervals(altered, epochs[1]), intervals)


def test_half_open_window_counts_duplicates_and_zero():
    times = np.array([0, .01, .01, .02, .03])
    windows = np.array([[0, .02], [.01, .03], [.04, .05]])
    np.testing.assert_array_equal(count_windows(times, windows), [3, 3, 0])
    np.testing.assert_array_equal(histogram_recount(times, windows), [3, 3, 0])
    with pytest.raises(ValueError, match="clock"):
        count_windows(times[::-1], windows)


def test_acquisition_tail_needs_independent_clock_evidence(tmp_path):
    root = tmp_path / "webshare_processed"
    limit, info = acquisition_clock_limit(root, "a", "s", 10., 9.5)
    assert limit == 10 and info["status"] == "all_spikes_within_declared_epochs"
    with pytest.raises(FileNotFoundError):
        acquisition_clock_limit(root, "a", "s", 10., 11.)
    folder = tmp_path / "raw_eeg" / "a" / "s"
    folder.mkdir(parents=True)
    (folder / "s.xml").write_text("<parameters><acquisitionSystem><nChannels>2</nChannels>"
        "<nBits>16</nBits><samplingRate>20000</samplingRate></acquisitionSystem>"
        "<fieldPotentials><lfpSamplingRate>10</lfpSamplingRate></fieldPotentials></parameters>")
    (folder / "s.eeg").write_bytes(bytes(480))
    limit, info = acquisition_clock_limit(root, "a", "s", 10., 11.5)
    assert limit == pytest.approx(12.00005)
    assert info["analysis_post_end_s"] == 10
    assert info["acquisition_duration_s"] == 12
    with pytest.raises(ValueError, match="outside"):
        acquisition_clock_limit(root, "a", "s", 10., 13.)


def test_endpoint_grid_not_peak_or_recency_selection():
    event = pd.DataFrame([dict(event_index=9, start_s=100., end_s=100.103, peak_s=100.01)])
    windows = windows_for(event)
    endpoint = windows.loc[windows.window.eq("endpoint")].iloc[0]
    assert endpoint.end_s == pytest.approx(100.100)
    assert endpoint.start_s == pytest.approx(100.080)
    event["peak_s"] = 100.001
    modified = windows_for(event)
    pd.testing.assert_frame_equal(modified.iloc[[0]], windows.iloc[[0]])
    assert not modified.iloc[1].available


def test_partition_equal_disjoint_deterministic_and_small_fail():
    groups, dropped = partition(23, 3, 44)
    assert [len(g) for g in groups] == [7, 7, 7]
    assert len(np.unique(np.concatenate(groups+[dropped]))) == 23
    assert len(dropped) == 2
    for a, b in zip(groups, partition(23, 3, 44)[0], strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="five"):
        partition(14, 3, 3)


def test_group_support_cannot_count_unavailable_window():
    windows = pd.DataFrame(dict(event_index=[0, 0], window=["endpoint", "peak_diagnostic"],
                                start_s=[1., 1.1], end_s=[1.02, 1.12], available=[True, False]))
    frame, parts, failures = group_readouts(np.ones((2, 18), int), windows, np.arange(18), "a", "s", 42)
    assert len(parts) == 6 and not failures
    assert frame.loc[frame.window.eq("endpoint"), "all_groups_supported"].all()
    assert not frame.loc[frame.window.eq("peak_diagnostic"), "all_groups_supported"].any()
    silent, _, _ = group_readouts(np.zeros((2, 18), int), windows, np.arange(18), "a", "s", 42)
    assert not silent.all_groups_supported.any()
    assert silent.loc[silent.window.eq("endpoint"), "all_groups_silent"].all()


def test_all_empty_or_unknown_denominators_fail():
    sessions = pd.DataFrame([dict(session=s, animal=s.split("_")[0], status="failed",
                                 candidates=np.nan, independent_recount_passed=False) for s in COHORT])
    _, _, gates = summarize(sessions, pd.DataFrame())
    assert not gates.passed.any()
    sessions["status"] = "complete"
    sessions["candidates"] = 100
    sessions["independent_recount_passed"] = True
    by_session, animals, gates = summarize(sessions, pd.DataFrame())
    assert by_session.all_groups_supported_fraction.eq(0).all()
    assert animals.all_groups_supported_fraction.eq(0).all()
    assert not gates.iloc[-1].passed


def test_native_fixture_complete_uid_clock_counts_and_first_half(tmp_path):
    name = "Achilles_10252013"
    folder = tmp_path / "native" / "Achilles" / name
    folder.mkdir(parents=True)
    ids = np.arange(1, 19)
    times = np.empty(len(ids), object)
    for i in range(len(ids)):
        # 30 first-half RUN spikes; POST bursts must not enter the unit screen.
        run = np.linspace(10.1, 19.8, 30)
        bursts = np.concatenate([peak + np.linspace(-.032, .032, 8) + i*.00002 for peak in [40., 60., 80.]])
        times[i] = np.r_[run, bursts]
    savemat(folder / f"{name}.spikes.cellinfo.mat", dict(spikes=dict(
        UID=ids, times=times, region=np.full(len(ids), "CA1", object), shankID=np.ones(len(ids)))))
    savemat(folder / f"{name}.CellClass.cellinfo.mat", dict(CellClass=dict(
        UID=ids[::-1], pE=np.ones(len(ids)), pI=np.zeros(len(ids)))))
    savemat(folder / f"{name}.position.behavior.mat", dict(position=source_position()))
    out = tmp_path / "out"
    out.mkdir()
    row, frame = extract_one(tmp_path / "native", name, out, 42)
    assert row["status"] == "complete"
    assert row["activity_screen_units"] == 18
    assert row["candidates"] == 3
    assert row["independent_recount_passed"]
    unit = pd.read_csv(Path(row["artifact_dir"]) / "native_unit_audit.csv")
    assert unit.first_half_run_spikes.eq(30).all()
    assert frame.loc[frame.window.eq("peak_diagnostic"), "all_groups_supported"].all()
    with np.load(Path(row["artifact_dir"]) / "count_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["epochs_s"][1, 0] == 10
        assert arrays["position_cm"][-1, 1] > 390
    audit, _ = audit_session(next(pd.DataFrame([row]).itertuples(index=False)), 42)
    assert audit["status"] == "passed"
    assert audit["window_cell_counts"] == 108
    target = Path(row["artifact_dir"])
    frame.loc[0, "group0_spikes"] += 1
    frame.to_csv(target / "population_counts.csv.gz", index=False)
    meta = json.loads((target / "manifest.json").read_text())
    meta["output_sha256"]["population_counts.csv.gz"] = file_sha256(target / "population_counts.csv.gz")
    (target / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        audit_session(next(pd.DataFrame([row]).itertuples(index=False)), 42)
