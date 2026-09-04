"""Native clocks, candidate denominators, and encoding-only selection contracts."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.replay_coverage_data import (
    TANNI_GENERAL,
    TANNI_POSITION,
    TANNI_SPIKES,
    CoverageInputConfig,
    align_tanni_clusters,
    array_sha256,
    canonical_candidates,
    count_candidate_bins,
    fit_coverage_population,
    load_tanni_coverage_session,
    prepare_position_support,
)
from scripts.prepare_replay_coverage_inputs import export_session, session_records


def make_session(position, spikes=None):
    position = np.asarray(position, dtype=float)
    spikes = np.asarray(spikes, dtype=float).reshape(-1, 2) if spikes is not None else np.empty((0, 2))
    empty = np.empty((0, 2))
    return ReplaySession(
        rat="R1", name="S1", path=Path("S1"), position=position, spikes=spikes,
        tetrode_cell_ids=empty, excitatory_neurons=np.unique(spikes[:, 1]).astype(int),
        inhibitory_neurons=np.empty(0, dtype=int), ripple_events=np.empty((0, 6)),
        run_times=np.array([[np.nanmin(position[:, 0]), np.nanmax(position[:, 0])]]),
        sleep_box_immobile_times=empty, sleep_times=empty, rem_times=empty,
        well_sequence=None, metadata={"source_dataset": "tanni2022"},
    )


@pytest.mark.parametrize("labels", [[1, 2, 9, 3, 2], [1, 2, 3, 2]])
def test_cluster_keep_applies_to_both_native_layouts(labels):
    times, cells = align_tanni_clusters(np.arange(5.0), labels, [1, 1, 0, 1, 1])
    np.testing.assert_array_equal(times, [1, 3, 4])
    np.testing.assert_array_equal(cells, [2, 3, 2])


@pytest.mark.parametrize("labels,keep", [([2, 2], None), ([2, 2], [1, 0, 0]), ([2, 2, 2], [1, 7, 1]), ([2.5, 2, 3], None)])
def test_cluster_misalignment_fails(labels, keep):
    with pytest.raises(ValueError):
        align_tanni_clusters([1, 2, 3], labels, keep)


def test_native_nwb_reads_clock_without_lfp_and_hashes_consumed_content(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "S1" / "experiment_1.nwb"
    path.parent.mkdir()
    with h5py.File(path, "w") as handle:
        handle.create_dataset(TANNI_POSITION, data=np.column_stack((np.arange(100, 101, .05), np.ones(20), np.ones(20))))
        handle.create_dataset(f"{TANNI_GENERAL}/arena_size", data=[100.0, 80.0])
        handle.create_dataset(f"{TANNI_GENERAL}/animal", data=np.bytes_("R1"))
        for electrode, times in [(1, [100.1, 100.4, 100.7]), (2, [100.2, 100.3, 100.8])]:
            group = f"{TANNI_SPIKES}/electrode{electrode}"
            handle.create_dataset(f"{group}/timestamps", data=times)
            handle.create_dataset(f"{group}/clustering/manual_1", data=[2, 3])
            handle.create_dataset(f"{group}/idx_keep", data=[True, False, True])
    session, meta = load_tanni_coverage_session(path)
    np.testing.assert_allclose(session.spikes[:, 0], [100.1, 100.2, 100.7, 100.8])
    np.testing.assert_array_equal(session.spikes[:, 1], [1002, 2002, 1003, 2003])
    assert session.position[0, 0] == 100
    assert session.rat == "R1"
    assert len(meta["consumed_datasets"]) == 9
    assert meta["hash_scope"] == "consumed_HDF5_datasets_not_entire_NWB"
    assert session.excitatory_neurons.size == 0


def test_array_hash_includes_shape_dtype_and_supports_native_strings():
    assert array_sha256(np.arange(4)) != array_sha256(np.arange(4).reshape(2, 2))
    assert array_sha256(np.arange(4)) != array_sha256(np.arange(4, dtype=float))
    assert array_sha256(np.array([b"rat"], dtype=object)) == array_sha256(np.array(["rat"], dtype=object))
    assert len(array_sha256(np.empty(0))) == 64
    with pytest.raises(ValueError):
        array_sha256(np.array([object()], dtype=object))


def test_half_open_candidate_counts_preserve_partial_tail_and_silence():
    spikes = np.array([[10.0, 1], [10.005, 1], [10.01, 2], [10.012, 1], [9.999, 1]])
    edges, counts = count_candidate_bins(spikes, [1, 2, 3], 10.0, 10.012, .005)
    np.testing.assert_allclose(np.diff(edges), [.005, .005, .002])
    np.testing.assert_array_equal(counts, [[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    _, later = count_candidate_bins(spikes, [1, 2, 3], 10.012, 10.022, .005)
    assert counts.sum() + later.sum() == 4
    _, silence = count_candidate_bins(spikes, [1, 2], 11, 11.05, .005)
    assert silence.shape == (10, 2) and silence.sum() == 0


def test_gap_frames_cannot_contribute_run_occupancy_or_interpolated_spikes():
    t = np.r_[np.arange(0, .5, .05), np.arange(2, 2.5, .05)]
    position = np.column_stack((t, np.arange(len(t)), np.ones(len(t))))
    result, qc = prepare_position_support(make_session(position), .1)
    assert len(result.run_times) == 2
    assert result.run_times[0, 1] < .45
    assert result.run_times[1, 0] > 2
    assert qc["position_samples_valid"] == len(t)
    assert not np.any((result.run_times[:, 0] <= 1) & (result.run_times[:, 1] >= 1))


def test_invalid_position_and_clock_reset_not_sorted_or_silently_bridged():
    t = np.r_[np.arange(0, .5, .05), .2, .4, .5, .55, .6, .65]
    position = np.column_stack((t, 20 * t, np.ones(len(t))))
    position[5, 1] = 1000
    result, qc = prepare_position_support(make_session(position), .1, np.array([[0, 0], [50, 50]]))
    assert qc["position_clock_rejected"] == 2
    assert qc["position_samples_valid"] == len(t) - 3
    assert np.all(np.diff(result.position[:, 0]) > 0)
    assert not np.any((result.run_times[:, 0] <= .25) & (result.run_times[:, 1] >= .25))


def tanni_events():
    return pd.DataFrame({
        "animal": ["R1"] * 3, "session": ["S1"] * 3,
        "mua_method": ["pooled_spike_density", "pooled_spike_density", "cellwise_z_mean"],
        "mua_event_index": [0, 1, 0], "core_start_time_s": [1., 2., 1.],
        "core_end_time_s": [1.1, 2.1, 1.1], "peak_time_s": [1.05, 2.05, 1.05],
        "continuity_pass": [True, False, True], "log_evidence": [100, -100, 2],
    })


def test_no_filter_on_continuity_or_model_evidence():
    frame = tanni_events()
    first = canonical_candidates(frame, "tanni2022")
    frame["continuity_pass"] = ~frame.continuity_pass
    frame["log_evidence"] *= -1000
    pd.testing.assert_frame_equal(first, canonical_candidates(frame, "tanni2022"))
    assert first.event_index.tolist() == [0, 1]
    assert "continuity_pass" not in first


def test_duplicate_candidate_ids_fail():
    with pytest.raises(ValueError, match="duplicate"):
        canonical_candidates(pd.concat([tanni_events(), tanni_events()]), "tanni2022")


def test_missing_manifest_pair_and_denominator_mismatch_fail():
    pf = pd.DataFrame(columns=["animal", "session", "event_index", "event_start_s", "event_end_s", "event_peak_s", "event_definition"])
    manifest = pd.DataFrame({"animal": ["R1"], "session": ["S1"], "nwb_path": ["S1/experiment_1.nwb"], "mua_candidates": [1]})
    with pytest.raises(ValueError, match="denominator"):
        session_records(Path("PF"), pf, manifest, tanni_events())
    manifest["session"] = "S2"
    with pytest.raises(ValueError, match="missing"):
        session_records(Path("PF"), pf, manifest, tanni_events())


def test_population_fit_not_changed_by_immobile_candidate_spikes():
    t = np.arange(0, 20, .02)
    x = np.where(t < 18, np.mod(t, 4) * 20, 20)
    position = np.column_stack((t, x, 10 + np.zeros(len(t))))
    spikes = np.array([(time, 2 + int(time % 4)) for time in np.arange(.1, 17.8, .03)])
    session, _ = prepare_position_support(make_session(position, spikes), .1)
    config = CoverageInputConfig(min_running_spikes=1, max_running_rate_hz=100)
    before, qc, _ = fit_coverage_population(session, config)
    altered = replace(session, spikes=np.vstack([session.spikes, [[19.0, 2]] * 100]))
    after, later_qc, _ = fit_coverage_population(altered, config)
    np.testing.assert_array_equal(before["rates_hz"], after["rates_hz"])
    pd.testing.assert_frame_equal(qc, later_qc)


def test_export_retains_zero_spike_events_and_consistent_offsets(tmp_path, monkeypatch):
    t = np.arange(0, 5, .02)
    session = make_session(np.column_stack((t, 20 * t, np.ones(len(t)))), [[.2, 2], [.3, 2], [1.05, 2]])
    monkeypatch.setattr("scripts.prepare_replay_coverage_inputs.load_tanni_coverage_session", lambda path: (session, {"arena_bounds_cm": [[0, 0], [120, 10]], "arena_bounds_source": "fixture"}))
    out = tmp_path / "out"
    (out / "sessions").mkdir(parents=True)
    candidates = canonical_candidates(tanni_events(), "tanni2022")
    events, _, summary = export_session("tanni2022", "R1", "S1", Path("S1"), candidates, out, CoverageInputConfig())
    assert len(events) == 2
    assert events.n_spikes_all_sorted.tolist() == [1, 0]
    assert summary["candidates"] == 2
    with np.load(events.cache_path.iloc[0], allow_pickle=False) as cache:
        offsets = cache["candidate_offsets"]
        assert offsets[0] == 0 and len(offsets) == 3
        assert offsets[-1] == cache["candidate_base_counts"].shape[0]
        np.testing.assert_array_equal(cache["candidate_event_indices"], [0, 1])
