import json

import numpy as np
import pandas as pd
import pytest
from hipporeplayimm.replay_coverage_detector_sensitivity import (
    condition_cohorts,
    nonoverlap_steps,
    paired_metrics,
    session_summary,
    summarize_animals,
)
from scripts._provenance import file_sha256
from scripts.analyze_replay_coverage_detector_sensitivity import input_arrays, process_session
from scripts.audit_replay_coverage_detector_sensitivity import audit_session, independent_metrics
from scripts.inspect_replay_coverage_lfp_mua_traces import select_examples
from scripts.report_replay_coverage_detector_sensitivity import paired_interactions


def source_windows(available=True):
    rows = []
    for detector in ["source_high_mua", "lfp_ripple_detected"]:
        for i, variant in enumerate(["detected_core", "peak_centered_200ms"]):
            rows.append({"dataset": "tanni2022", "animal": "R", "session": "day", "window_uid": f"{detector}:{variant}",
                "detector": detector, "window_variant": variant, "source_event_id": i, "eligible": available or detector == "source_high_mua",
                "start_s": 0., "end_s": .083 if i == 0 else .2, "peak_s": .04, "ripple_peak_z": 4.5,
                "ripple_status": "available" if available else "unavailable_insufficient_baseline",
                "overlap_class": "both_detectors" if available else "ripple_unavailable", "n_spikes_qc_units": 10})
    return pd.DataFrame(rows)


def test_speed_requires_all_intermediate_frames():
    path = np.column_stack([np.arange(12) * 4, np.zeros(12)])
    support = np.ones(12, bool)
    support[2] = False
    indices, speed, valid = nonoverlap_steps(path, support)
    np.testing.assert_array_equal(indices, [0, 4, 8])
    np.testing.assert_allclose(speed, [800, 800])
    assert valid.tolist() == [False, True]
    assert support[indices].all(), "endpoint-only policy would incorrectly retain first step"


def test_paired_speed_uses_identical_surviving_steps():
    full = np.column_stack([np.arange(12) * 4, np.zeros(12)])
    path = full * 2
    counts = np.full((12, 2), 2)
    subset = counts.copy()
    subset[2] = 0
    m = paired_metrics(path, subset, np.ones(12), np.ones(12), full, counts, True)
    assert m["common_measurable_steps"] == 1
    assert m["median_common_step_speed_delta_cm_s"] == 800
    assert m["paired_continuity_delta"] <= 0


def test_empty_short_events_are_failures_not_missing_rows():
    m = paired_metrics(np.empty((0, 2)), np.empty((0, 1)), [], [], np.empty((0, 2)), np.empty((0, 2)), True)
    assert not m["continuity_pass"] and m["common_measurable_steps"] == 0
    assert np.isnan(m["median_event_speed_cm_s"])


def test_common_speed_rejects_different_clocks():
    with pytest.raises(ValueError, match="identical time"):
        paired_metrics(np.ones((10, 2)), np.ones((10, 2)), np.ones(10), np.ones(10), np.ones((11, 2)), np.ones((11, 3)), False)


def test_unavailable_cohorts_are_not_zero_ripple_observations():
    cohorts = list(condition_cohorts(source_windows(False), "tanni2022"))
    for c, ids, available in cohorts:
        if c["detector"] == "lfp_ripple_detected":
            assert len(ids) == 0 and not available
        if c["detector"] == "source_high_mua" and c["overlap_scope"] == "ripple_unavailable":
            assert len(ids) == 1 and available


def fixture(tmp_path):
    cache = tmp_path / "cache.npz"
    ids = np.array([2, 4, 6, 8])
    spikes = np.array([[.005, 2], [.020, 4], [.045, 6], [.08, 8], [.09, 4], [.2, 2]])
    np.savez(cache, cell_ids=ids, spikes=spikes, unit_qc_mask=np.ones(4, bool),
        rates_hz=np.array([[10, 2, 1], [1, 10, 2], [2, 1, 10], [5, 2, 5]]),
        valid_spatial_bins=np.ones(3, bool), bin_centers_cm=np.array([[0., 0], [40, 0], [80, 0]]),
        arena_bounds_cm=np.array([[0, 0], [100, 100]]))
    windows = source_windows()
    wp = tmp_path / "windows.csv"
    windows.to_csv(wp, index=False)
    return {"dataset": "tanni2022", "animal": "R", "session": "day", "source_cache_path": str(cache),
        "source_cache_sha256": file_sha256(cache), "windows_path": str(wp), "windows_sha256": file_sha256(wp)}, windows, spikes


def test_count_windows_preserve_event_edges(tmp_path):
    record, windows, spikes = fixture(tmp_path)
    arrays = input_arrays(record, windows)
    for k, (start, end) in enumerate(zip(arrays["frame_start_s"], arrays["frame_end_s"], strict=True)):
        np.testing.assert_allclose(end - start, .02)
        raw = spikes[(spikes[:, 0] >= start) & (spikes[:, 0] < end), 1]
        np.testing.assert_array_equal(arrays["frame_counts"][k], [np.count_nonzero(raw == cell) for cell in arrays["cell_ids"]])
    assert np.diff(arrays["frame_offsets"]).tolist() == [13, 37, 13, 37]


def test_full_pipeline_condition_counts_and_paired_reference(tmp_path):
    record, windows, _ = fixture(tmp_path)
    out = tmp_path / "output"
    out.mkdir()
    meta = process_session(record, out, 20260905, 3)
    assert meta["metric_rows"] == 4 * 32
    metrics = pd.read_csv(meta["metrics_path"])
    full = metrics[metrics.cell_fraction.eq(1.)]
    assert full.paired_continuity_delta.eq(0).all()
    assert full.median_common_step_speed_delta_cm_s.dropna().eq(0).all()
    for path in meta["decoded_paths"]:
        assert file_sha256(path["path"]) == path["sha256"]
    population = pd.read_csv(meta["population_path"])
    session, animal, summary = summarize_animals(population, 10)
    assert session.recording_subsets[session.cell_fraction.eq(.5)].eq(3).all()
    assert len(animal) > 0 and summary.animals_total.eq(1).all()
    with pytest.raises(ValueError, match="missing/duplicated"):
        session_summary(metrics.iloc[1:], windows, meta["population_specs"])
    assert json.loads(next(out.glob("session__*.json")).read_text())["status"] == "complete"
    checked = audit_session(meta, 20260905, 3)
    assert checked["independent_metric_rows"] == 128 and checked["analytic_posterior_frames"] > 0


def test_zero_eligible_session_remains_explicit(tmp_path):
    record, windows, _ = fixture(tmp_path)
    windows.eligible = False
    windows.to_csv(record["windows_path"], index=False)
    record["windows_sha256"] = file_sha256(record["windows_path"])
    out = tmp_path / "output"
    out.mkdir()
    meta = process_session(record, out, 1, 1)
    assert meta["eligible_windows"] == 0 and meta["metric_rows"] == 0
    pop = pd.read_csv(meta["population_path"])
    assert pop.eligible_events.dropna().eq(0).all()
    assert pop.continuity_fraction.isna().all()
    assert audit_session(meta, 1, 1)["directly_counted_frames"] == 0


def test_independent_metrics_cover_gaps_ties_and_both_estimators():
    rng = np.random.default_rng(8)
    for n in [0, 1, 4, 5, 10, 40]:
        path = rng.normal(0, 8, (n, 2)).cumsum(axis=0)
        full = np.rint(path / 8) * 8
        full_counts = rng.poisson(2, (n, 5))
        counts = full_counts[:, :2]
        for supported in [False, True]:
            args = (path, counts, np.ones(n), np.ones(n), full, full_counts, supported)
            expected = paired_metrics(*args)
            audited = independent_metrics(*args)
            assert set(audited) == set(expected)
            np.testing.assert_allclose([audited[k] for k in expected], list(expected.values()), equal_nan=True)


def test_trace_selection_is_stable_and_ignores_decoding_scores():
    frame = source_windows()
    frame = pd.concat([frame.assign(source_event_id=i, n_spikes_qc_units=i) for i in range(5)], ignore_index=True)
    first = select_examples(frame)
    second = select_examples(frame.sample(frac=1, random_state=6).assign(log_evidence=1e10, continuity_pass=True))
    assert first.source_event_id.tolist() == [2]
    assert second.source_event_id.tolist() == first.source_event_id.tolist()


def test_interactions_pair_sessions_before_averaging_animals():
    rows = []
    for animal, sessions in [("A", 3), ("B", 1)]:
        for day in range(sessions):
            for detector in ["source_high_mua", "lfp_ripple_detected"]:
                for variant in ["detected_core", "peak_centered_200ms"]:
                    change = -.2 if detector == "source_high_mua" else (-.1 if animal == "A" else -.3)
                    rows.append({"dataset": "tanni2022", "animal": animal, "session": str(day),
                        "detector": detector, "window_variant": variant, "likelihood": "poisson", "estimator": "map",
                        "bin_filter": "unfiltered", "cell_fraction": .5, "overlap_scope": "all_eligible",
                        "peak_threshold_z": 3 if detector == "lfp_ripple_detected" else 0,
                        "paired_continuity_delta": change, "median_common_step_speed_delta_cm_s": change * 100})
    _, animals, summary = paired_interactions(pd.DataFrame(rows).sample(frac=1, random_state=2), 50)
    row = summary[(summary.contrast == "Ripple core minus MUA core") & (summary.metric == "paired_continuity_delta")].iloc[0]
    assert abs(row.equal_animal_mean) < 1e-12, "three A sessions must not outweigh one B session"
    assert row.animals_measurable == 2 and row.paired_sessions == 4
    assert animals.paired_sessions.max() == 3
