import json

import numpy as np
import pandas as pd
import pytest

from scripts._provenance import file_sha256
from scripts.denovellis_post_error_core import build_transitions, reconstruct_visits, score_visits, valid_intervals
from scripts.denovellis_post_error_neural import make_graph
from scripts.denovellis_post_error_v2 import FROZEN, restrict_transition
from scripts.denovellis_post_error_v2_readout import (
    UnavailableReadout,
    cohort_after_qc,
    decode_middle_third,
    fit_first_third,
    summarize_windows,
    training_samples,
    validation_windows,
)
from scripts.verify_denovellis_post_error_v2 import independent_trials, verify_readout

P = json.loads(FROZEN.read_text())


def graph_fixture():
    return make_graph([[0, 0, 0, 50], [0, 50, -30, 50], [-30, 50, -30, 0], [0, 50, 30, 50], [30, 50, 30, 0]], [[0, 0], [-30, 0], [30, 0]], 1, (2, 3))


def simulation():
    graph = graph_fixture()
    xy = np.tile(graph.xy, (12, 1))
    t = np.arange(len(xy)) * 0.02
    speed = np.full(len(t), 10.0)
    bounds = np.linspace(t[0], t[-1], 4)
    features = np.column_stack([xy, xy])
    marks = {1: (t + 0.001, features.copy()), 2: (t + 0.001, features.copy())}
    return graph, t, xy, speed, bounds, marks


def test_no_future_positions_speeds_or_marks_can_change_encoding():
    graph, t, xy, speed, b, marks = simulation()
    traversals = [{"start_s": 0, "end_s": b[1]}]
    fit, _, _ = fit_first_third(graph, t, xy, speed, marks, b, traversals, P)
    xy2, v2 = xy.copy(), speed.copy()
    xy2[t >= b[1]], v2[t >= b[1]] = 10000, 1e9
    marks2 = {k: (mt, np.where((mt >= b[1])[:, None], mf + 10000, mf)) for k, (mt, mf) in marks.items()}
    changed, _, _ = fit_first_third(graph, t, xy2, v2, marks2, b, traversals, P)
    np.testing.assert_array_equal(fit.occupancy, changed.occupancy)
    for tet in fit.features:
        np.testing.assert_array_equal(fit.features[tet], changed.features[tet])
        np.testing.assert_array_equal(fit.spatial[tet], changed.spatial[tet])


def test_training_tracking_edges_and_whole_traversals():
    t = np.arange(0, 3, 0.1)
    xy, v = np.zeros((len(t), 2)), np.ones(len(t)) * 10
    b = [0, 1, 2, 3]
    tt, _, vv = training_samples(t, xy, v, b, [{"start_s": 0.25, "end_s": 0.75}])
    assert tt[-1] < 1
    assert np.all(tt[np.isfinite(vv)] >= 0.25)
    assert np.all(tt[np.isfinite(vv)] + 0.1 <= 0.75)
    with pytest.raises(UnavailableReadout):
        training_samples(t, xy, v, b, [])


def test_validation_disallows_gap_anywhere_in_window():
    graph = graph_fixture()
    t = np.r_[np.arange(0, 1.011, 0.001), np.arange(1.4, 3.001, 0.001)]
    xy = np.tile([-30.0, 20.0], (len(t), 1))
    a, z, _, _ = validation_windows(graph, t, xy, np.ones(len(t)) * 10, [0, 1, 2, 3], [{"start_s": 1, "end_s": 2}], 0.02)
    assert not np.any((a < 1.4) & (z > 1.01))
    assert np.all(a >= 1) and np.all(z <= 2)


def test_silent_validation_bins_are_scored_not_filtered():
    graph, t, xy, v, b, marks = simulation()
    fit, _, _ = fit_first_third(graph, t, xy, v, marks, b, [{"start_s": 0, "end_s": b[1]}], P)
    silent = {tet: (np.empty(0), np.empty((0, 4))) for tet in marks}
    qc, windows = decode_middle_third(fit, t, xy, v, silent, b, [{"start_s": b[1], "end_s": b[2]}], P)
    assert qc["n_arm_windows"] == qc["zero_spike_windows"] > 0
    assert not qc["decoder_qc_passed"]
    assert set(windows.true_arm) == {0, 1}


def test_missing_arm_is_unavailable_not_low_accuracy():
    frame = pd.DataFrame({"true_arm": [0, 0], "predicted_arm": [0, 0], "n_spikes": [0, 2], "n_active_tetrodes": [0, 2]})
    with pytest.raises(UnavailableReadout, match="validation_support"):
        summarize_windows(frame, P)


def test_metrics_independently_match_confusion_counts():
    frame = pd.DataFrame({"true_arm": [0] * 8 + [1] * 4, "predicted_arm": [0] * 7 + [1] + [1] * 3 + [0], "n_spikes": range(12), "n_active_tetrodes": [2] * 12})
    qc = summarize_windows(frame, P)
    assert qc["arm0_recall"] == 7 / 8 and qc["arm1_recall"] == 3 / 4
    assert qc["balanced_accuracy"] == (7 / 8 + 3 / 4) / 2
    assert qc["decoder_qc_passed"]


def test_cohort_cannot_pass_from_one_day_or_only_corrections():
    rows = [
        {"animal": a, "day": d, "session": f"{a}-{d}", "final_third_eligible": True, "next_outcome": o} for a in "abcde" for d in (1, 2) for o in ("correction", "repeated_error")
    ]
    trials = pd.DataFrame(rows)
    qc = pd.DataFrame(
        [
            {
                "session": f"{a}-{d}",
                "decoder_qc_passed": d == 1,
                "status": "passed" if d == 1 else "failed_accuracy",
                "balanced_accuracy": 0.9 if d == 1 else 0.6,
                "arm0_recall": 0.9 if d == 1 else 0.6,
                "arm1_recall": 0.9 if d == 1 else 0.6,
            }
            for a in "abcde"
            for d in (1, 2)
        ]
    )
    cohort, animals, gates = cohort_after_qc(trials, qc, "abcde", P)
    assert cohort.decoder_qualified_transition.sum() == 10
    assert not cohort.primary_cohort_eligible.any() and not animals.animal_supported.any()
    assert not any(g["passed"] for g in gates)
    qc.loc[0, "balanced_accuracy"] = np.nan
    with pytest.raises(ValueError, match="numerical"):
        cohort_after_qc(trials, qc, "abcde", P)


@pytest.mark.parametrize("tracking_gap", [False, True])
@pytest.mark.parametrize("immobile_speed", [0.0, 4.0])
def test_independent_raw_trial_reconstruction(tracking_gap, immobile_speed):
    wells = np.array([[0.0, 0], [-40.0, 0], [40.0, 0]])
    t = np.arange(0, 90.1, 0.1)
    xy = np.tile([0.0, 20.0], (len(t), 1))
    for start, well in [(61, 2), (64, 1), (67, 2), (70, 1), (73, 2), (76, 1), (79, 3)]:
        xy[(t >= start) & (t < start + 2)] = wells[well - 1]
    if tracking_gap:
        xy[(t > 69.1) & (t < 69.5)] = np.nan
    speed = np.ones(len(t)) * immobile_speed
    raw, _, _ = reconstruct_visits(t, xy, wells)
    visits = score_visits(raw, 1, (2, 3))
    near = np.linalg.norm(xy - wells[0], axis=1) <= P["well_radius_cm"]
    intervals = valid_intervals(t, near & (speed < 4), 0.25)
    legacy = build_transitions(visits, center=1, outers=(2, 3), immobile_intervals=intervals, min_exposure=0.5, max_window=10)
    actual = [restrict_transition(r, visits, [0, 30, 60, 90]) for r in legacy]
    independent, verified_visits = independent_trials(t, xy, speed, wells, 1, (2, 3), P)
    assert [v["visit_index"] for v in verified_visits] == list(range(1, len(verified_visits) + 1))
    assert len(actual) == len(independent) == (1 if tracking_gap else 2)
    for a, b in zip(actual, independent, strict=True):
        assert a["final_third_eligible"] == b["eligible"]
        assert a["next_outcome"] == b["outcome"]
        assert a["usable_exposure_s"] == pytest.approx(b["exposure_s"])


def verification_fixture(tmp_path):
    prefix = "denovellis_post_error_"
    audit, out = tmp_path / "audit", tmp_path / "readout"
    audit.mkdir()
    out.mkdir()
    (out / "checkpoints").mkdir()
    (audit / (prefix + "manifest.json")).write_text(json.dumps({"protocol": P, "audit_passed": True, "output_sha256": {}}))
    trials = pd.DataFrame(
        [
            {
                "session": f"a-{day}",
                "animal": "a",
                "day": day,
                "final_third_eligible": True,
                "decoder_qualified_transition": False,
                "primary_cohort_eligible": False,
                "next_outcome": o,
            }
            for day in (1, 2)
            for o in ("correction", "repeated_error")
        ]
    )
    qc = []
    for day in (1, 2):
        session = f"a-{day}"
        qc.append(
            {
                "session": session,
                "status": "failed_accuracy",
                "reason": "",
                "decoder_qc_passed": False,
                "train_end_s": 1,
                "validation_end_s": 2,
                "balanced_accuracy": 0.5,
                "arm0_recall": 1.0,
                "arm1_recall": 0.0,
                "n_arm_windows": 4,
                "zero_spike_windows": 1,
            }
        )
        windows = pd.DataFrame(
            {
                "start_time_s": [1.0, 1.02, 1.04, 1.06],
                "end_time_s": [1.02, 1.04, 1.06, 1.08],
                "true_arm": [0, 0, 1, 1],
                "predicted_arm": [0] * 4,
                "n_spikes": [0, 1, 1, 1],
                "arm0_mass": [0.6] * 4,
                "arm1_mass": [0.3] * 4,
            }
        )
        windows.to_csv(out / "checkpoints" / (session + "_windows.csv"), index=False)
    for name, frame in [
        ("trial_inventory", trials),
        ("decoder_audit", pd.DataFrame(qc)),
        ("input_inventory", pd.DataFrame(columns=["path", "sha256"])),
        ("by_animal", pd.DataFrame([{"animal": "a", "transitions": 0, "corrections": 0, "repetitions": 0, "recording_days": 0, "animal_supported": False}])),
        ("gate_summary", pd.DataFrame([{"observed": 0, "passed": False}] * 4)),
    ]:
        frame.to_csv(out / (prefix + name + ".csv"), index=False)
    manifest = {
        "protocol": P,
        "source_directory": str(audit),
        "source_manifest_sha256": file_sha256(audit / (prefix + "manifest.json")),
        "feasibility_passed": False,
        "calibration_passed": False,
        "biological_status": "not_tested",
        "output_sha256": {f.name: file_sha256(f) for f in out.glob("*.csv")},
        "checkpoint_sha256": {f.name: file_sha256(f) for f in (out / "checkpoints").glob("*.csv")},
    }
    (out / (prefix + "manifest.json")).write_text(json.dumps(manifest))
    return out


def test_independent_readout_verifier_recomputes_accuracy(tmp_path):
    out = verification_fixture(tmp_path)
    result = verify_readout(out, out / "verification.json")
    assert result["status"] == "verified" and result["n_validation_windows"] == 8
    assert result["n_passing_epochs"] == result["n_primary_transitions"] == 0


def test_verifier_detects_metric_error_even_with_updated_file_hash(tmp_path):
    out = verification_fixture(tmp_path)
    path = out / "denovellis_post_error_decoder_audit.csv"
    metrics = pd.read_csv(path)
    metrics.loc[0, "arm0_recall"] = 0.99
    metrics.to_csv(path, index=False)
    manifest_path = out / "denovellis_post_error_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"][path.name] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="arm0_recall"):
        verify_readout(out, out / "verification.json")
