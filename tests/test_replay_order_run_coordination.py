"""Measurement invariants, not a validation of a biological association."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import h5py

from scripts import audit_replay_order_run_coordination as audit
from scripts import verify_replay_pair_measurements as verifier
from scripts.verify_replay_pair_measurements import reference_order


@pytest.fixture
def protocol():
    path = Path(__file__).resolve().parents[1] / "docs/replay_order_run_coordination_protocol.json"
    return json.loads(path.read_text())


def test_whole_bin_shuffles_preserve_population_vectors():
    c = np.array([[3, 0, 1], [0, 2, 0], [1, 0, 0], [0, 0, 0]])
    nulls = audit.whole_bin_shuffles(c, 20, 13)
    for n in nulls:
        np.testing.assert_array_equal(n.sum(axis=0), c.sum(axis=0))
        assert sorted(map(tuple, n)) == sorted(map(tuple, c))
    for a, b in zip(nulls, audit.whole_bin_shuffles(c, 20, 13), strict=True):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(c, n) for n in nulls)


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
def test_invalid_shuffle_counts(n):
    with pytest.raises(ValueError):
        audit.whole_bin_shuffles(np.ones((4, 2)), n, 1)


def test_order_sign_and_reverse():
    c = np.array([[2, 0], [0, 0], [0, 3]])
    original = audit.order_asymmetry(c, .005, .005, .06)
    assert original[0, 1] == 1
    np.testing.assert_array_equal(original, -original.T)
    np.testing.assert_allclose(audit.order_asymmetry(c[::-1], .005, .005, .06), -original)


def test_independent_raw_pair_enumeration_matches_matrix_order():
    c = np.random.default_rng(13).poisson(.7, (14, 6))
    original = audit.order_asymmetry(c, .0047, .005, .06)
    for i in range(6):
        for j in range(i + 1, 6):
            a = np.repeat(np.arange(len(c)), c[:, i])
            b = np.repeat(np.arange(len(c)), c[:, j])
            assert reference_order(a, b, .0047, .005, .06) == pytest.approx(original[i, j])


def test_simultaneous_and_silent_cells_have_zero_direction():
    c = np.array([[1, 2, 0], [0, 0, 0], [1, 2, 0]])
    np.testing.assert_array_equal(audit.order_asymmetry(c, .005, .005, .06), np.zeros((3, 3)))


def test_order_shuffling_removes_injected_asymmetry_in_expectation():
    c = np.zeros((20, 2), dtype=int)
    c[3, 0], c[8, 1] = 4, 4
    scores = [audit.order_asymmetry(n, .005, .005, .06)[0, 1]
              for n in audit.whole_bin_shuffles(c, 1000, 19)]
    assert audit.order_asymmetry(c, .005, .005, .06)[0, 1] == 1
    assert abs(np.mean(scores)) < .08


def test_complete_duration_equal_bins_and_half_open_spikes():
    spikes = np.array([[1., 1], [1.001, 2], [1.052, 1], [1.053, 2]])
    counts, width = audit.event_counts(spikes, np.array([1, 2]), 1., 1.053, .005)
    assert len(counts) == 11
    assert width * len(counts) == pytest.approx(.053)
    np.testing.assert_array_equal(counts.sum(axis=0), [2, 1])


def test_stable_identity_seed_not_traversal_order():
    assert audit.stable_seed(9, "Rat1/Open1:native:3") == audit.stable_seed(9, "Rat1/Open1:native:3")
    assert audit.stable_seed(9, "Rat1/Open1:native:3") != audit.stable_seed(9, "Rat2/Open1:native:3")


@pytest.mark.parametrize("time", [[0, .02, .01], [0, 0, .02], [0, np.nan, .02]])
def test_chronology_is_not_sorted_away(protocol, time):
    position = np.column_stack((time, np.zeros((3, 2))))
    with pytest.raises(ValueError):
        audit.epoch_links(position, np.array([[0, 2]]), protocol)


def test_tracking_gap_interrupts_pause(protocol):
    p = {**protocol, "pause_min_duration_s": .05}
    pos = np.array([[0, 0, 0], [.05, 0, 0], [.1, 0, 0], [1, 0, 0], [1.05, 0, 0], [1.1, 0, 0]])
    links = audit.epoch_links(pos, np.array([[0, 2]]), p)
    pauses = audit.immobile_pauses(links, p)
    assert [(x["start_s"], x["end_s"]) for x in pauses] == [(0., .1), (1., 1.1)]


def test_epoch_boundary_interrupts_pause(protocol):
    p = {**protocol, "pause_min_duration_s": .05}
    pos = np.column_stack((np.arange(0, .21, .05), np.zeros((5, 2))))
    links = audit.epoch_links(pos, np.array([[0, .1], [.1, .2]]), p)
    pauses = audit.immobile_pauses(links, p)
    assert len(pauses) == 2
    assert pauses[0]["epoch_index"] != pauses[1]["epoch_index"]


def test_overlapping_epochs_rejected(protocol):
    pos = np.array([[0, 0, 0], [.02, 0, 0], [.04, 0, 0]])
    with pytest.raises(ValueError, match="Overlapping"):
        audit.epoch_links(pos, np.array([[0, 1], [0, 2]]), protocol)


def test_matching_requires_position_direction_and_speed(protocol):
    links = {"t": np.array([0., 1., 2., 3., 4., 5.]), "dt": np.ones(5),
             "xy": np.array([[8., 8.], [8., 8.], [0., 0.], [8., 8.], [8., 8.], [0., 0.]]),
             "speed": np.array([15., 15., 0., 15., 15.]),
             "direction": np.array([0., np.pi, 0., 0., 0.]),
             "good": np.ones(5, dtype=bool), "epoch": np.zeros(5, dtype=int)}
    pause = {"start_s": 2., "end_s": 3., "epoch_index": 0}
    m, before, after = audit.matched_run(links, pause, protocol)
    assert m["common_strata"] == 1
    assert m["matched_exposure_s"] == 1
    np.testing.assert_array_equal(np.flatnonzero(before), [0])
    np.testing.assert_array_equal(np.flatnonzero(after), [3, 4])
    changed = {**links, "speed": np.array([15., 15., 0., 45., 45.])}
    assert audit.matched_run(changed, pause, protocol)[0]["common_strata"] == 0


def test_spikes_on_tracking_gap_not_assigned(protocol):
    pos = np.array([[0, 0, 0], [.05, 1, 0], [1., 2, 0], [1.05, 3, 0]])
    links = audit.epoch_links(pos, np.array([[0, 2]]), protocol)
    times = np.array([-.1, 0., .03, .5, 1., 1.05])
    np.testing.assert_array_equal(audit.link_spikes(times, links, links["good"]),
                                  [False, True, True, False, True, False])


def synthetic_session():
    times = np.arange(0, 41.001, .025)
    x = np.zeros(len(times))
    for mask, offset in [(times < 20, 0), (times >= 21, 21)]:
        phase = np.mod(times[mask] - offset, 4)
        x[mask] = np.where(phase < 2, phase, 4 - phase) * 15
    position = np.column_stack((times, x, np.zeros(len(times))))
    spikes = np.array([(t, c) for t in np.arange(.1, 19.9, .1) for c in (1, 2)] +
                      [(20.1, 1), (20.11, 1), (20.12, 1), (20.15, 2), (20.16, 2), (20.17, 2)])
    events = np.array([[20.09, 20.2, 20.15, 1, 2, 3], [20.8, 21.1, 21., 1, 2, 3]])
    return SimpleNamespace(position=position, run_times=np.array([[0, 42]]), spikes=spikes,
                           excitatory_spikes=lambda: spikes, excitatory_neurons=np.array([1, 2]),
                           ripple_events=events, session_id="Rat1/Open1", rat="Rat1",
                           ripple=lambda i: SimpleNamespace(start=events[i, 0], end=events[i, 1]))


def test_future_spikes_do_not_select_units_and_events_must_be_contained(protocol, monkeypatch, tmp_path):
    p = {**protocol, "minimum_eligible_units": 2, "minimum_matched_run_exposure_s": .1,
         "candidate_min_active_units": 2, "candidate_min_spikes": 2}
    session = synthetic_session()
    monkeypatch.setattr(audit, "load_replay_session", lambda _: session)
    monkeypatch.setattr(audit, "variable_inventory", lambda _: [])
    before = audit.audit_session(tmp_path, p)
    assert len(before[1]) == 1
    assert before[1][0]["order_measured"]
    assert before[0][0]["eligible_preceding_units"] == 2
    session.spikes = np.vstack((session.spikes, np.array([[22, 99], [23, 99]])))
    session.excitatory_spikes = lambda: session.spikes
    after = audit.audit_session(tmp_path, p)
    assert before[0] == after[0]
    assert before[1] == after[1]
    assert before[2] == after[2]


def test_missing_theta_and_unvalidated_candidates_do_not_complete_goal(protocol):
    assert protocol["require_theta_phase"] is True
    assert protocol["primary_analysis_enabled"] is False
    assert "not_validated_replay" in protocol["candidate_label"]
    assert "lfp_theta_phase" in protocol["mandatory_controls"]


def test_tanni_lfp_header_does_not_claim_clock_alignment(tmp_path):
    path = tmp_path / "experiment_1.nwb"
    with h5py.File(path, "w") as handle:
        r = handle.create_group("acquisition/timeseries/recording1")
        r.create_dataset("tracking/ProcessedPos", data=np.array([[1, 0, 0], [2, 1, 1]]))
        r.create_dataset("continuous/processor102_100/downsampled_tetrode_data", data=np.zeros((4, 2)))
        r.create_dataset("continuous/processor102_100/downsampled_timestamps", data=[0, 1, 2, 3])
        r.create_dataset("continuous/processor102_100/downsampling_info/downsampled_sampling_rate", data=1500)
    rows = audit.tanni_lfp_headers(tmp_path)
    assert len(rows) == 1
    assert rows[0]["tracking_within_lfp_clock_bounds"]
    assert rows[0]["clock_alignment_verified"] is False
    assert rows[0]["matched_run_replication_tested"] is False
    assert rows[0]["integrity_scope"] == "header_and_endpoint_reads_not_full_file_checksum"


def test_audit_cli_never_turns_empty_measurements_into_pass(protocol, monkeypatch, tmp_path):
    protocol["sessions"] = ["Rat1/Open1"]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    summary = {"animal": "Rat1", "session": "Rat1/Open1", "immobile_pauses": 0,
               "matched_run_supported_pauses": 0, "nonoverlapping_supported_pauses": 0,
               "order_measured_events": 0, "candidate_pair_rows": 0}
    monkeypatch.setattr(audit, "audit_session", lambda *_: ([], [], [], [], summary))
    assert audit.main(["--dataset-root", str(tmp_path), "--protocol", str(path),
                       "--output-dir", str(tmp_path / "output")]) == 0
    import pandas as pd
    gates = pd.read_csv(tmp_path / "output/gate_summary.csv")
    assert not gates["passed"].any()
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["status"] == "partial_measurement_audit_missing_pf_theta_not_biological_test"
    with pytest.raises(ValueError, match="empty"):
        audit.main(["--dataset-root", str(tmp_path), "--protocol", str(path),
                    "--output-dir", str(tmp_path / "output")])


@pytest.fixture
def verified_fixture(protocol, monkeypatch, tmp_path):
    p = {**protocol, "sessions": ["Rat1/Open1"], "minimum_eligible_units": 2,
         "minimum_matched_run_exposure_s": .1, "candidate_min_active_units": 2,
         "candidate_min_spikes": 2}
    dataset = tmp_path / "dataset"
    session_path = dataset / "Rat1/Open1"
    session_path.mkdir(parents=True)
    source = session_path / "Spike_Data.mat"
    source.write_text("synthetic source identity")
    session = synthetic_session()
    monkeypatch.setattr(audit, "load_replay_session", lambda _: session)
    monkeypatch.setattr(verifier, "load_replay_session", lambda _: session)
    monkeypatch.setattr(audit, "variable_inventory", lambda _: [
        {"path": str(source), "sha256": audit.file_sha256(source), "variables": []}])
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(p))
    root = tmp_path / "measurement"
    audit.main(["--dataset-root", str(dataset), "--protocol", str(path), "--output-dir", str(root)])
    return root, dataset


def refresh_output_hash(root, name):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs"][name] = audit.file_sha256(root / name)
    path.write_text(json.dumps(manifest))


def test_independent_verifier_reconciles_all_originals_not_biology(verified_fixture):
    root, dataset = verified_fixture
    result = verifier.verify(root, dataset)
    assert result["original_pair_scores_independently_reconstructed"] == 1
    assert result["measured_events"] == 1
    assert result["sessions_reconciled"] == 1
    assert result["contained_candidates_reconciled"] == 1
    assert result["max_absolute_original_order_error"] < 1e-12
    assert result["theta_control_verified"] is False
    assert result["biological_association_verified"] is False


def test_verifier_checks_the_actual_loaded_copy(verified_fixture, tmp_path):
    root, _ = verified_fixture
    dataset = tmp_path / "different_copy"
    folder = dataset / "Rat1/Open1"
    folder.mkdir(parents=True)
    (folder / "Spike_Data.mat").write_text("a different raw source")
    with pytest.raises(ValueError, match="Loaded dataset hash"):
        verifier.verify(root, dataset)


def test_verifier_rejects_unhashed_extra_source(verified_fixture):
    root, dataset = verified_fixture
    (dataset / "Rat1/Open1/Extra_Marks.mat").write_text("unhashed optional source")
    with pytest.raises(ValueError, match="file inventory"):
        verifier.verify(root, dataset)


@pytest.mark.parametrize("problem", ["empty_pairs", "altered_order", "missing_candidate"])
def test_verifier_rejects_invalid_tables_even_if_hashes_updated(verified_fixture, problem):
    import pandas as pd
    root, dataset = verified_fixture
    name = "candidate_event_inventory.csv" if problem == "missing_candidate" else "candidate_pair_order.csv"
    frame = pd.read_csv(root / name)
    if problem in ("empty_pairs", "missing_candidate"):
        frame = frame.iloc[:0]
    else:
        frame.loc[0, "a_before_b_asymmetry"] += .1
    frame.to_csv(root / name, index=False)
    refresh_output_hash(root, name)
    with pytest.raises(ValueError):
        verifier.verify(root, dataset)
