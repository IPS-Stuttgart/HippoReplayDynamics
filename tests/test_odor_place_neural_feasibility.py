import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts import odor_place_neural_feasibility as neural
from scripts.odor_place_feasibility_core import PREFIX, read_table
from scripts.odor_place_source_io import atomic_json

PROTOCOL = json.loads((Path(__file__).parents[1] / "docs/odor_place_source_v3_neural_feasibility_protocol.json").read_text())


def graph():
    return neural.graph_grid([[[0, 30], [0, 0], [-30, 0]], [[0, 30], [0, 0], [30, 0]]], 3.)


def test_graph_distance_never_smooths_across_unconnected_arm_locations():
    g = graph()
    assert np.allclose(np.diag(g["distance"]), 0)
    assert np.allclose(g["distance"], g["distance"].T)
    left, right = np.flatnonzero(g["edges"] == 1)[-1], np.flatnonzero(g["edges"] == 2)[-1]
    assert g["distance"][left, right] == 57.
    assert neural.project_bins([[-29, 0], [29, 0], [0, 29]], g).tolist() == [19, 29, 0]


def test_chronological_split_does_not_use_future_traversals_or_outcomes():
    traversals = [{"start_s": i * 2., "end_s": i * 2. + 1, "arm": "left" if i % 2 == 0 else "right"} for i in range(20)]
    a, b = neural.chronological_split(traversals, 28., PROTOCOL["run"])
    assert max(r["end_s"] for r in a) < min(r["start_s"] for r in b)
    altered = [{**r, "outcome": "future_changed"} for r in traversals]
    assert neural.chronological_split(altered, 28., PROTOCOL["run"])[0][-1]["end_s"] == a[-1]["end_s"]
    with pytest.raises(ValueError, match="Insufficient"):
        neural.chronological_split(traversals, 5., PROTOCOL["run"])
    with pytest.raises(ValueError, match="overlap"):
        neural.chronological_split(traversals[:14] + [traversals[0]], 28., PROTOCOL["run"])


def test_zero_spike_decoding_has_flat_prior_but_informative_poisson_silence():
    fitted = {"rates": np.array([[1.], [10.], [100.]]), "support": np.array([True, True, False]), "selected_units": np.array([0])}
    posterior = neural.decode(np.zeros((1, 1)), fitted, .02)[0]
    expected = np.exp(-.02 * np.array([1., 10.]))
    expected /= expected.sum()
    assert np.allclose(posterior[:2], expected)
    assert posterior[2] == 0


def test_run_windows_retain_zero_spikes_reject_tracking_gaps_and_future_spikes():
    times = np.arange(0., 2., .02)
    data = np.column_stack([times, -times * 10, np.zeros(len(times)), np.zeros(len(times)), np.full(len(times), 10.)])
    traversals = [{"start_s": .2, "end_s": 1.2, "arm": "left"}]
    original = neural.run_windows(data, traversals, [np.array([])], graph(), PROTOCOL["run"], .1)[0]
    changed = neural.run_windows(data, traversals, [np.array([1.3, 4.])], graph(), PROTOCOL["run"], .1)[0]
    assert len(original["counts"]) > 0
    assert np.array_equal(original["counts"], changed["counts"])
    assert not original["counts"].any()
    gapped = data[(times < .4) | (times > .8)]
    reduced = neural.run_windows(gapped, traversals, [np.array([])], graph(), PROTOCOL["run"], .1)[0]
    assert len(reduced["counts"]) < len(original["counts"])
    assert not any(a < .8 and b > .4 for a, b in zip(reduced["start"], reduced["end"], strict=True))


def test_metrics_balance_traversals_and_include_unsupported_truth():
    g = graph()
    rates = np.ones((len(g["xy"]), 1))
    fitted = {"rates": rates, "support": g["edges"] == 1, "selected_units": np.array([0])}
    windows = [{"truth": np.full(n, truth), "counts": np.zeros((n, 1), int), "arm": arm}
               for n, truth, arm in [(100, 11, "left"), (1, 12, "left"), (1, 21, "right")]]
    metrics = neural.validation_metrics(windows, fitted, g, PROTOCOL["run"])
    assert metrics["balanced_accuracy"] == .5
    assert metrics["right_recall"] == 0
    assert metrics["support_coverage"] == pytest.approx(2 / 3)
    assert metrics["n_zero_spike_bins"] == 102
    assert not neural.metrics_pass(metrics, PROTOCOL["run"])


def test_encoding_filters_units_on_training_counts_only():
    g = graph()
    truth = np.tile(np.arange(len(g["xy"])), 30)
    counts = np.column_stack([np.ones(len(truth), int)] * 5 + [np.zeros(len(truth), int)])
    windows = [{"truth": truth, "counts": counts, "arm": "left"}]
    fitted = neural.fit_maps(windows, g, PROTOCOL["run"])
    assert fitted["selected_units"].tolist() == list(range(5))
    assert np.isfinite(fitted["rates"]).all()
    assert fitted["support"].all()
    assert np.allclose(neural.decode(counts, fitted, .02).sum(axis=1), 1)


def test_ambiguous_unit_spike_identity_or_electrode_semantics_never_qualify():
    source = {"spikes": np.array([1., 2.]), "area": "CA1", "tetrode": 3, "cluster": 2, "tag": "unit"}
    refs = [[{"area": "CA1", "tetrode": 3}]]
    rows, spikes, _ = neural.verify_unit_identity([10], [source["spikes"]], refs, [source])
    assert rows[0]["status"] == "verified_ca1_spike_identity" and len(spikes) == 1
    assert neural.verify_unit_identity([10], [source["spikes"]], refs, [source, dict(source)])[0][0]["status"] == "excluded"
    refs[0].append({"area": "PFC", "tetrode": 4})
    assert not neural.verify_unit_identity([10], [source["spikes"]], refs, [source])[1]


def test_nwb_electrode_id_and_row_index_ambiguity_is_explicit(tmp_path):
    path = tmp_path / "test.nwb"
    with h5py.File(path, "w") as f:
        u = f.create_group("units")
        u["id"], u["electrodes"] = [12], [1]
        u["spike_times"], u["spike_times_index"] = [1., 2.], [2]
        e = f.create_group("general/extracellular_ephys/electrodes")
        e["id"] = [1, 9]
        e["location"] = np.array(["CA1", "PFC"], dtype=h5py.string_dtype())
        e["group_name"] = np.array(["tetrode3", "tetrode4"], dtype=h5py.string_dtype())
        source = [{"spikes": np.array([1., 2.]), "area": "CA1", "tetrode": 3, "cluster": 2, "tag": "unit"}]
        rows, spikes, _ = neural.nwb_units(f, source, (0, 3))
        assert not spikes and rows[0]["failure_reason"] == "electrode_identity_conflicts_or_is_ambiguous"


def test_ripple_opportunities_require_whole_containment_and_keep_zero_events():
    cfg = PROTOCOL["ripples"]
    events = [[1., 1.12], [1.95, 2.1], [3., 3.2]]
    spikes = [np.arange(1.005, 1.115, .02), np.array([1.035, 1.085])]
    rows = neural.opportunity_counts(events, [[1., 2.]], spikes, cfg, .02)
    assert len(rows) == 1 and rows[0]["sequence_testing_opportunity"]
    assert neural.opportunity_counts(events, [[4., 5.]], spikes, cfg, .02) == []
    assert neural.merge_overlap([[1, 2], [1.5, 3], [3.01, 4]]) == [[1., 3.], [3.01, 4.]]


def test_ripple_kernel_rejects_gaps_and_inadequate_sampling():
    cfg = PROTOCOL["ripples"]
    with pytest.raises(ValueError, match="sampling"):
        neural.detect_ripples(np.sin(np.arange(100)), np.arange(100) / 400, cfg)
    times = np.arange(2000) / 1000
    times[1000:] += .1
    with pytest.raises(ValueError, match="gapped"):
        neural.detect_ripples(np.sin(np.arange(2000)), times, cfg)
    times = np.arange(5000) / 1000
    signal = np.random.default_rng(20261001).normal(0, .05, len(times))
    signal[(times > 2) & (times < 2.15)] += 5 * np.sin(2 * np.pi * 200 * times[(times > 2) & (times < 2.15)])
    a = neural.detect_ripples(signal, times, cfg)
    assert a == neural.detect_ripples(signal, times, cfg)
    assert any(start < 2.1 < stop for start, stop in a)


@pytest.mark.parametrize("stage", ["acquire-neural", "run-qc"])
def test_failed_source_gate_prevents_all_neural_downloads_and_reads(tmp_path, monkeypatch, stage):
    from scripts import odor_place_original_source_audit as source
    atomic_json(tmp_path / (PREFIX + "decision.json"), {"source_screen_passed": False, "n_eligible_source_transitions": 136})
    monkeypatch.setattr(source, "verify", lambda *_: {"raw_source_reconciliation": {"status": "verified_against_raw_digital_edges"}})
    monkeypatch.setattr(neural, "selected_assets", lambda *_: pytest.fail("No neural selection after source failure"))
    result = neural.dispatch(SimpleNamespace(stage=stage, output_dir=tmp_path), PROTOCOL)
    assert result["status"] == "inconclusive_source_feasibility"
    assert not result["neural_processing_performed"]
    assert result["full_nwb_files_downloaded"] == 0
    assert read_table(tmp_path, "run_validation") == []


def test_provisional_novelty_does_not_become_a_pass(tmp_path, monkeypatch):
    from scripts import odor_place_original_source_audit as source
    atomic_json(tmp_path / (PREFIX + "decision.json"), {"source_screen_passed": True, "n_eligible_source_transitions": 200})
    monkeypatch.setattr(source, "verify", lambda *_: {"raw_source_reconciliation": {"status": "verified_against_raw_digital_edges"}})
    result = neural.dispatch(SimpleNamespace(stage="acquire-neural", output_dir=tmp_path), PROTOCOL)
    assert result["failure_reason"] == "novelty_prerequisite_not_passed"


def test_qualified_zero_event_trials_are_not_dropped_from_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(neural, "independently_verify_neural", lambda *_: atomic_json(tmp_path / (PREFIX + "neural_verification.json"), {"status": "coverage_fixture"}))
    atomic_json(tmp_path / (PREFIX + "inventory_identity.json"), {"output_sha256": {}})
    atomic_json(tmp_path / (PREFIX + "neural_acquisition.json"), {"complete": True, "assets": []})
    protocol_path = tmp_path / "protocol.json"
    atomic_json(protocol_path, PROTOCOL)
    args = SimpleNamespace(output_dir=tmp_path, protocol=protocol_path)
    errors, validation, opportunities = [], [], []
    for animal in PROTOCOL["full_maze_animals_source_code"]:
        for condition in ["repeat", "changed"]:
            for outcome in ["correction_to_prior_cue_arm", "repeated_mistaken_arm"]:
                for i in range(5):
                    key = f"{animal}:{condition}:{outcome}:{i}"
                    errors.append({"animal": animal, "source_trial_key": key, "cue_condition": condition, "outcome_relative_to_prior_error_cue": outcome})
                    validation.append({"source_trial_key": key, "decoder_qualified": True, "status": "evaluated"})
                    opportunities.append({"source_trial_key": key, "status": "counted_not_validated_replay", "candidate_count": 0, "sequence_testing_opportunities": 0})
    result = neural.finalize_neural(args, PROTOCOL, errors, validation, opportunities)
    assert result["ready_for_calibration"] and result["n_qualified_zero_event_transitions"] == 100
    assert result["n_supported_ripple_opportunities"] == 0
    assert not neural.finalize_neural(args, PROTOCOL, [], [], [])["ready_for_calibration"]


def test_incomplete_or_duplicated_neural_rows_cannot_pass(tmp_path):
    errors = [{"source_trial_key": "trial"}]
    with pytest.raises(ValueError, match="exactly one"):
        neural.finalize_neural(SimpleNamespace(output_dir=tmp_path), PROTOCOL, errors, [], [])


def test_independent_likelihood_verifier_detects_changed_metrics(tmp_path):
    g = graph()
    truth = np.tile(np.arange(len(g["xy"])), 30)
    train = [{"truth": truth, "counts": np.ones((len(truth), 5), int), "arm": "left"}]
    fitted = neural.fit_maps(train, g, PROTOCOL["run"])
    test = [{"truth": np.full(5, 11), "counts": np.zeros((5, 5), int), "arm": "left"},
            {"truth": np.full(5, 21), "counts": np.zeros((5, 5), int), "arm": "right"}]
    metrics = neural.validation_metrics(test, fitted, g, PROTOCOL["run"])
    (tmp_path / "neural_checkpoints").mkdir()
    np.savez_compressed(tmp_path / "neural_checkpoints/trial.npz", **fitted, edges=g["edges"],
                        validation_counts=np.concatenate([w["counts"] for w in test]),
                        validation_truth=np.concatenate([w["truth"] for w in test]),
                        traversal_lengths=[5, 5], traversal_arms=["left", "right"])
    row = {"source_trial_key": "trial", "status": "evaluated", **metrics}
    neural.independently_verify_neural(SimpleNamespace(output_dir=tmp_path), PROTOCOL, [row])
    with pytest.raises(ValueError, match="Independent"):
        neural.independently_verify_neural(SimpleNamespace(output_dir=tmp_path), PROTOCOL, [{**row, "balanced_accuracy": 1.}])
