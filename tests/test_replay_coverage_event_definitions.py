from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.replay_coverage_event_definitions import (
    EventDefinitionConfig,
    InsufficientBaselineError,
    annotate_overlap,
    detect_ripple_episodes,
    immobile_intervals,
    interval_contains,
    make_windows,
    overlap_pairs,
    ripple_envelope,
    ripple_window_metrics,
    source_events,
    standardize_envelope,
    tracking_speed,
    validate_lfp_clock,
    window_spike_support,
)
from scripts.prepare_replay_coverage_event_definitions import load_ripple_envelope, native_ripples, summarize_windows


def sources():
    mua = pd.DataFrame({"event_index": [1, 2], "start_s": [1., 1.4], "end_s": [1.2, 1.6], "peak_s": [1.1, 1.5]})
    ripple = pd.DataFrame({"source_event_id": [0, 1], "start_s": [1.1, 1.15], "end_s": [1.45, 1.25], "peak_s": [1.2, 1.2], "ripple_peak_z": [4., 5.], "detector_duration_pass": [True, True]})
    return source_events(mua, ripple, "tanni2022", "rat", "day")


def test_speed_never_bridges_position_or_run_gaps():
    t = np.r_[np.arange(0, 1, .02), np.arange(2, 3, .02)]
    xy = np.column_stack([np.where(t < 1, 0, 100), np.zeros(len(t))])
    speed = tracking_speed(np.column_stack([t, xy]), [[0, 3]], EventDefinitionConfig())
    np.testing.assert_allclose(speed, 0, atol=1e-10)
    intervals = immobile_intervals(t, speed, EventDefinitionConfig())
    assert len(intervals) == 2
    assert not interval_contains([.5], [2.5], intervals)[0]
    assert interval_contains([.5], [.6], intervals)[0]
    speed2 = tracking_speed(np.column_stack([t, xy]), [[0, .5], [2.5, 2.98]], EventDefinitionConfig())
    assert np.isnan(speed2[(t > .5) & (t < 2.5)]).all()


def test_immobility_linear_threshold_crossing():
    config = replace(EventDefinitionConfig(), maximum_position_gap_s=2)
    np.testing.assert_allclose(immobile_intervals([0, 1, 2], [0, 10, 0], config), [[0, .5], [1.5, 2]])


def test_irregular_lfp_fails_without_clock_repair():
    t = np.arange(1000) / 1500
    validate_lfp_clock(t, 1500)
    t[500:] += .01
    with pytest.raises(ValueError, match="irregular"):
        validate_lfp_clock(t, 1500)


def test_episode_merge_duration_and_exclusions():
    fs = 1000
    t, z = np.arange(2000) / fs, np.full(2000, -1.)
    z[100:150], z[170:200], z[500:900], z[1200:1250] = 4., 6., 5., 2.
    out = detect_ripple_episodes(t, z, fs, EventDefinitionConfig())
    assert len(out) == 2
    np.testing.assert_allclose(out.start_s, [.1, .5])
    np.testing.assert_allclose(out.end_s, [.2, .9])
    assert out.detector_duration_pass.tolist() == [True, False]
    assert out.ripple_peak_z.tolist() == [6., 5.]


def test_missing_lfp_samples_do_not_merge():
    t, z = np.arange(1000) / 1000, np.full(1000, -1.)
    z[100:150], z[160:200], z[150:160] = 5, 4, np.nan
    assert len(detect_ripple_episodes(t, z, 1000, EventDefinitionConfig())) == 2


def test_standardization_uses_only_immobile_supported_baseline():
    t = np.arange(10000) / 1000
    envelope = 2 + .1 * np.sin(20 * t)
    envelope[t > 5] += 100
    config = replace(EventDefinitionConfig(), minimum_baseline_s=1)
    z, baseline, meta = standardize_envelope(envelope, t, 1000, [[1, 4]], config)
    np.testing.assert_allclose(z[baseline].mean(), 0, atol=1e-10)
    np.testing.assert_allclose(z[baseline].std(), 1)
    assert np.isnan(z[t < 1]).all()
    assert meta["baseline_duration_s"] == pytest.approx(3.001)
    assert np.nanmean(z[t > 6]) > 100
    with pytest.raises(ValueError, match="insufficient"):
        standardize_envelope(envelope, t, 1000, [[1, 1.2]], config)
    with pytest.raises(InsufficientBaselineError) as exc:
        standardize_envelope(envelope, t, 1000, [[1, 1.2]], config)
    assert exc.value.duration_s == pytest.approx(.201)


def test_bandpass_envelope_prefers_ripple_band():
    fs = 1500
    t = np.arange(7500) / fs
    c = EventDefinitionConfig()
    ripple = ripple_envelope(np.sin(2 * np.pi * 200 * t), fs, c)
    slow = ripple_envelope(np.sin(2 * np.pi * 30 * t), fs, c)
    assert np.mean(ripple[1500:-1500]) > 100 * np.mean(slow[1500:-1500])
    with pytest.raises(ValueError, match="constant"):
        ripple_envelope(np.zeros(100), fs, c)


def test_two_window_variants_keep_all_exclusions_no_clipping():
    events = sources()
    p = np.array([[0., 0, 0], [2., 0, 0]])
    out = make_windows(events, p, [[0, 2]], [[1.05, 1.55]], EventDefinitionConfig())
    assert len(out) == 8 and out.window_uid.is_unique
    assert out.groupby("source_uid").size().eq(2).all()
    assert out.loc[~out.eligible, "exclusion_reason"].str.len().gt(0).all()
    fixed = out[out.window_variant.eq("peak_centered_200ms")]
    np.testing.assert_allclose(fixed.window_duration_s, .2)
    assert out.peak_immobile.all()
    assert not out.whole_window_immobile.all()


def test_many_to_many_overlap_never_duplicates_events():
    windows = make_windows(sources(), np.array([[0., 0, 0], [2., 0, 0]]), [[0, 2]], [[0, 2]], EventDefinitionConfig())
    pairs = overlap_pairs(windows)
    core = pairs[pairs.window_variant.eq("detected_core")]
    assert len(core) == 3
    out = annotate_overlap(windows, pairs)
    assert len(out) == len(windows)
    first = out[out.window_variant.eq("detected_core") & out.detector.eq("source_high_mua")]
    assert first.other_detector_overlap_count.tolist() == [2, 1]


def test_touching_windows_do_not_overlap():
    rows = sources().iloc[[0, 2]].copy()
    rows.loc[rows.detector.eq("lfp_ripple_detected"), ["start_s", "end_s", "peak_s"]] = [1.2, 1.3, 1.25]
    w = make_windows(rows, np.array([[0., 0, 0], [2., 0, 0]]), [[0, 2]], [[0, 2]], EventDefinitionConfig())
    assert overlap_pairs(w[w.window_variant.eq("detected_core")]).empty


def test_invalid_identity_fails():
    e = sources()
    mua = pd.DataFrame({"event_index": [1, 1], "start_s": [1., 1.4], "end_s": [1.2, 1.6], "peak_s": [1.1, 1.5]})
    with pytest.raises(ValueError, match="duplicate"):
        source_events(mua, e[e.detector.eq("lfp_ripple_detected")], "tanni2022", "rat", "day")


def test_spike_support_half_open_boundaries_and_empty_units():
    w = pd.DataFrame({"start_s": [0., 1.], "end_s": [1., 2.]})
    metrics = window_spike_support(w, {1: np.array([0, .5, 1, 2]), 2: np.array([.1])}, [1, 2], [True, False])
    assert metrics.n_spikes_all_sorted.tolist() == [3, 1]
    assert metrics.n_spikes_qc_units.tolist() == [2, 1]
    assert metrics.n_active_qc_units.tolist() == [1, 1]


def test_ripple_metrics_missingness_explicit():
    w = pd.DataFrame({"start_s": [0., 1., 3.], "end_s": [1., 2., 4.]})
    metrics = ripple_window_metrics(w, np.arange(20) / 10, np.r_[np.full(10, np.nan), np.full(10, 4.)])
    assert metrics.ripple_samples_measured.tolist() == [0, 10, 0]
    assert np.isnan(metrics.mean_ripple_envelope_z.iloc[0])
    assert metrics.fraction_ripple_samples_above_z3.iloc[1] == 1


def test_zero_eligible_denominators_preserved_in_summary():
    w = make_windows(sources(), np.array([[0., 0, 0], [2., 0, 0]]), [[0, 2]], [[0, .1]], EventDefinitionConfig())
    w = annotate_overlap(w, overlap_pairs(w))
    w["n_spikes_qc_units"], w["n_active_qc_units"] = 0, 0
    summary = summarize_windows(w)
    assert len(summary) == 8
    assert summary.eligible_windows.eq(0).all()
    assert summary.n_spikes_qc_units_median.isna().all()


def test_unavailable_ripple_is_not_zero_or_mua_only():
    events = sources()
    events = events[events.detector.eq("source_high_mua")]
    w = make_windows(events, np.array([[0., 0, 0], [2., 0, 0]]), [[0, 2]], [[0, 2]], EventDefinitionConfig())
    w["ripple_status"] = "unavailable_insufficient_baseline"
    w = annotate_overlap(w, overlap_pairs(w))
    assert w.overlap_class.eq("ripple_unavailable").all()
    assert w.other_detector_overlap_count.isna().all()
    w["n_spikes_qc_units"], w["n_active_qc_units"] = 1, 1
    summary = summarize_windows(w)
    ripple = summary[summary.detector.eq("lfp_ripple_detected")]
    assert not ripple.detector_available.any()
    assert ripple.eligible_windows.isna().all()
    assert ripple.source_windows.isna().all()
    assert summary.other_detector_overlap_at_primary_z3.isna().all()


def test_native_tables_preserve_out_of_run_and_source_ids(tmp_path):
    from scipy.io import savemat
    a = np.array([[.1, .2, .15, 99, 77, 66], [10, 11, 10.5, 3, 4, 5]])
    savemat(tmp_path / "Ripple_Events.mat", {"Ripple_Events": a})
    events, meta = native_ripples(tmp_path)
    np.testing.assert_array_equal(events[["start_s", "end_s", "peak_s"]], a[:, :3])
    assert events.source_event_id.tolist() == [0, 1]
    assert meta["raw_lfp_available"] is False


def test_nwb_channel_selection_uses_literal_ca1_ids(tmp_path):
    import h5py

    from scripts.prepare_replay_coverage_event_definitions import CHANNEL_MAP, LFP_GROUP
    fs, path = 1500, tmp_path / "tiny.nwb"
    t = np.arange(4500) / fs
    wave = (1000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
    with h5py.File(path, "w") as f:
        g = f.require_group(LFP_GROUP)
        g["downsampled_timestamps"] = t
        g["downsampled_tetrode_data"] = np.column_stack([wave, wave * 2, wave * 0, np.full(len(t), 32767, np.int16)])
        g["downsampling_info/downsampled_sampling_rate"] = fs
        g["downsampling_info/downsampled_channels"] = [1, 4, 8, 12]
        f.require_group(CHANNEL_MAP)["CA1_LH/list"] = [1, 8, 12]
    times, envelope, rate, channels, meta = load_ripple_envelope(path, EventDefinitionConfig())
    np.testing.assert_array_equal(times, t)
    np.testing.assert_allclose(envelope, ripple_envelope(wave, fs, EventDefinitionConfig()))
    assert rate == fs and meta["channels_selected"] == 1
    assert [c["selected"] for c in channels] == [True, False, False, False]
    assert len(meta["consumed_datasets"]) == 5
    from scripts._provenance import file_sha256
    from scripts.audit_replay_coverage_event_definitions import verify_session
    from scripts.prepare_replay_coverage_event_definitions import prepare_session

    cache = tmp_path / "encoding.npz"
    pt = np.arange(0, 3, .02)
    np.savez(cache, position=np.column_stack([pt, np.zeros((len(pt), 2))]),
        supported_run_intervals=np.array([[0., 2.98]]), spikes=np.array([[1.06, 1], [1.08, 2]]),
        cell_ids=np.array([1, 2]), unit_qc_mask=np.array([True, True]))
    output = tmp_path / "out"
    (output / "sessions").mkdir(parents=True)
    record = {"dataset": "tanni2022", "animal": "Rat", "session": "day", "artifact_path": str(cache), "artifact_sha256": file_sha256(cache), "source_path": str(path)}
    mua = pd.DataFrame({"event_index": [1], "start_s": [1.], "end_s": [1.2], "peak_s": [1.1]})
    metadata, summary, _ = prepare_session(record, mua, output, EventDefinitionConfig())
    assert metadata["status"] == "complete" and metadata["ripple_status"] == "unavailable_insufficient_baseline"
    assert metadata["source_high_mua_events"] == 1 and metadata["source_ripple_events"] is None
    assert pd.DataFrame(summary).query("detector == 'lfp_ripple_detected'").eligible_windows.isna().all()
    assert verify_session(metadata, EventDefinitionConfig(), mua, refilter=True)["window_rows_verified"] == 2


@pytest.mark.parametrize("bad_hash", [False, True])
def test_cli_synthetic_session_manifest_and_failure_gates(tmp_path, bad_hash):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from scipy.io import savemat

    from scripts._provenance import file_sha256

    source = tmp_path / "raw"
    source.mkdir()
    savemat(source / "Ripple_Events.mat", {"Ripple_Events": np.array([[1.05, 1.15, 1.1], [10., 11., 10.5]])})
    cache = tmp_path / "encoding.npz"
    t = np.arange(0, 3, .02)
    np.savez(cache, position=np.column_stack([t, np.zeros((len(t), 2))]),
        supported_run_intervals=np.array([[0., 2.98]]), spikes=np.array([[1.06, 1], [1.08, 2]]),
        cell_ids=np.array([1, 2]), unit_qc_mask=np.array([True, True]))
    pd.DataFrame([{"dataset": "pfeiffer_foster", "animal": "Rat", "session": "Rat/Open", "source_path": str(source),
        "artifact_path": str(cache), "artifact_sha256": "bad" if bad_hash else file_sha256(cache), "status": "cached", "candidates": 1}]).to_csv(tmp_path / "coverage_input_sessions.csv", index=False)
    pd.DataFrame([{"dataset": "pfeiffer_foster", "animal": "Rat", "session": "Rat/Open", "event_index": 9,
        "start_s": 1., "end_s": 1.2, "peak_s": 1.1}]).to_csv(tmp_path / "coverage_input_candidates.csv", index=False)
    (tmp_path / "coverage_input_manifest.json").write_text("{}")
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_replay_coverage_event_definitions.py"
    completed = subprocess.run([sys.executable, str(script), "--input-dir", str(tmp_path), "--output-dir", str(out), "--workers", "1"], capture_output=True, text=True, check=False)
    assert (completed.returncode != 0) == bad_hash, completed.stdout + completed.stderr
    manifest = json.loads((out / "coverage_event_definition_manifest.json").read_text())
    assert manifest["status"] == ("failed" if bad_hash else "complete")
    gates = pd.read_csv(out / "coverage_event_definition_gate_summary.csv").set_index("gate").passed
    assert bool(gates["overall_preparation"]) is not bad_hash
    assert bool(gates["all_source_mua_events_retained"]) is not bad_hash
    if not bad_hash:
        windows = pd.read_csv(out / "coverage_event_definition_windows.csv")
        assert len(windows) == 6 and windows.eligible.sum() == 4
        assert windows.loc[~windows.eligible, "exclusion_reason"].str.contains("position_clock_valid").all()
        assert manifest["code_commit"] != "unavailable"
        for name, expected in manifest["output_sha256"].items():
            assert file_sha256(out / name) == expected
        auditor = script.with_name("audit_replay_coverage_event_definitions.py")
        audited = subprocess.run([sys.executable, str(auditor), "--input-dir", str(out), "--workers", "1"], capture_output=True, text=True, check=False)
        assert audited.returncode == 0, audited.stdout + audited.stderr
        assert json.loads((out / "coverage_event_definition_reconstruction_audit.json").read_text())["window_rows_verified"] == 6
        reporter = script.with_name("report_replay_coverage_event_definitions.py")
        reported = subprocess.run([sys.executable, str(reporter), "--input-dir", str(out), "--output-dir", str(tmp_path / "report")], capture_output=True, text=True, check=False)
        assert reported.returncode == 0, reported.stdout + reported.stderr
        assert (tmp_path / "report/coverage_event_definition_overview.png").stat().st_size > 1000
        from scripts.report_replay_coverage_event_definitions import summarize

        audit_path = out / "coverage_event_definition_reconstruction_audit.json"
        wrong = json.loads(audit_path.read_text())
        wrong["input_file_sha256"]["preparation_manifest"] = "other-run"
        audit_path.write_text(json.dumps(wrong))
        with pytest.raises(ValueError, match="does not cover"):
            summarize(out)
