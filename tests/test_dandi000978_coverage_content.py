"""Synthetic coverage/content tests; no private annotations or raw data."""

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts import dandi000978_coverage_content as m


def metadata_fixture():
    epochs = pd.DataFrame({"epoch_index": [0, 1, 2], "start_time_s": [0.0, 100.0, 200.0], "stop_time_s": [100.0, 200.0, 300.0], "n_trials": [0, 10, 0]})
    swr = pd.DataFrame({"startTime": [50.0, 150.0, 250.0, 270.0, 299.9], "endTime": [50.1, 150.1, 250.1, 270.01, 300.01], "epoch": [1, 2, 3, 3, 3], "source_row": np.arange(5)})
    nrem = pd.DataFrame({"startTime": [0.0, 200.0], "endTime": [100.0, 300.001], "epoch": [1, 3], "source_row": [0, 1]})
    return swr, nrem, epochs


def test_source_events_require_prior_run_rest_nrem_and_strict_boundaries():
    swr, nrem, epochs = metadata_fixture()
    result = m.candidate_metadata(swr, nrem, epochs, "JS14", "JS14.nwb").set_index("source_row")
    assert result.loc[0, "exclusion_reason"] == "no_preceding_RUN_in_file"
    assert result.loc[1, "exclusion_reason"] == "RUN_epoch"
    assert result.loc[2, "exclusion_reason"] == ""
    assert result.loc[3, "exclusion_reason"] == "duration"
    assert result.loc[4, "exclusion_reason"] == "invalid_or_outside_epoch"
    assert result.loc[2, "training_run_epoch"] == 1


def test_source_epoch_mismatch_is_not_silently_relabelled():
    swr, nrem, epochs = metadata_fixture()
    swr.loc[2, "epoch"] = 2
    with pytest.raises(ValueError, match="numbering"):
        m.candidate_metadata(swr, nrem, epochs, "JS14", "JS14.nwb")


def test_candidate_cap_is_seeded_support_is_ca1_only_and_windows_unchanged(monkeypatch):
    monkeypatch.setitem(m.PARAMETERS, "max_events_per_rest_epoch", 3)
    starts = 210 + np.arange(10) * 0.2
    metadata = pd.DataFrame(
        {
            "event_id": [str(i) for i in range(10)],
            "start_time_s": starts,
            "end_time_s": starts + 0.1,
            "source_row": np.arange(10),
            "rest_epoch": 2,
            "training_run_epoch": 1,
            "exclusion_reason": "",
        }
    )
    spikes = np.sort(np.r_[starts + 0.03, starts + 0.04])
    a = m.select_candidates(metadata, {1: [spikes] * 5}, "source")
    b = m.select_candidates(metadata, {1: [spikes] * 5}, "source")
    pd.testing.assert_frame_equal(a, b)
    assert a.selected.sum() == 3
    np.testing.assert_array_equal(a.start_time_s, starts)
    np.testing.assert_array_equal(a.end_time_s, starts + 0.1)
    assert (a.n_ca1_spikes == 10).all()
    assert "pfc" not in " ".join(a.columns).lower()


def test_overlapping_candidates_are_not_duplicated():
    meta = pd.DataFrame({"start_time_s": [1.0, 1.01], "end_time_s": [1.1, 1.11], "source_row": [0, 1], "training_run_epoch": 1, "rest_epoch": 2, "exclusion_reason": ""})
    result = m.select_candidates(meta, {1: [np.array([1.03, 1.04])] * 5}, "overlap")
    assert list(result.selected) == [True, False]
    assert result.exclusion_reason.iloc[1] == "overlapping_earlier_eligible_event"


def test_nested_coverage_is_constant_across_events_and_changes_across_repeats():
    a, b = m.nested_coverage(40, ("file", 1), 0), m.nested_coverage(40, ("file", 1), 1)
    assert {f: len(x) for f, x in a.items()} == {1.0: 40, 0.75: 30, 0.5: 20, 0.25: 10}
    assert set(a[0.25]) <= set(a[0.5]) <= set(a[0.75]) <= set(a[1.0])
    assert not np.array_equal(a[0.5], b[0.5])
    for f, ix in a.items():
        np.testing.assert_array_equal(ix, m.nested_coverage(40, ("file", 1), 0)[f])


def test_flat_prior_continuity_and_zero_spike_behavior():
    centers = np.column_stack([np.arange(24) * 4.0, np.zeros(24)])
    rates = np.full((24, 24), 0.1)
    np.fill_diagonal(rates, 100)
    windows = np.eye(24, dtype=int) * 5
    result = m.classify(windows, rates, centers, np.arange(24))
    assert result["edge_only"]["geometric_pass"]
    assert not result["bin_support"]["geometric_pass"]  # only one active cell per window
    silent = m.classify(np.zeros_like(windows), rates, centers, np.arange(24))
    assert not silent["edge_only"]["geometric_pass"]


def test_counts_obey_half_open_windows_and_no_short_bin_inflation():
    spikes = [np.array([0.0, 0.005, 0.019, 0.02, 0.027])]
    base = m.base_counts(spikes, 0.0, 0.027)
    assert len(base) == 5 and base.sum() == 4
    assert m.counts_in_interval(spikes, 0.0, 0.027)[0] == 4
    assert m.counts_in_interval(spikes, 0.027, 0.03)[0] == 1


def test_composition_removes_global_count_only_signal_and_keeps_zero_events():
    rates = np.array([np.arange(1, 6) * k for k in (1, 2, 3, 4)])
    scores, p = m.composition_scores(np.array([[0] * 5, [1, 2, 3, 4, 5]]), rates)
    np.testing.assert_allclose(scores, 0, atol=1e-14)
    np.testing.assert_allclose(p, 0.25, atol=1e-14)


def strong_fixture(n=48):
    targets = np.arange(n) % 4
    counts = np.zeros((n, 16), int)
    rates = np.ones((4, 16))
    for k in range(4):
        rates[k, k * 4 : (k + 1) * 4] = 30
        counts[targets == k, k * 4 : (k + 1) * 4] = 5
    vectors, _ = m.composition_scores(counts, rates)
    events = pd.DataFrame({"event_id": [str(i) for i in range(n)], "file": "synthetic", "rest_epoch": 2, "duration_s": 0.1, "n_pfc_spikes": 20, "ca1_reference_route": targets})
    return counts, rates, targets, vectors, events


def test_map_null_preserves_per_cell_rates_and_detects_synthetic_content(monkeypatch):
    monkeypatch.setitem(m.PARAMETERS, "null_draws", 39)
    counts, rates, targets, vectors, _ = strong_fixture()
    null, permutations = m.shuffled_map_scores(counts, rates, targets, "test")
    np.testing.assert_array_equal(np.sort(permutations, axis=1), np.broadcast_to(np.arange(4)[None, :, None], permutations.shape))
    real = vectors[np.arange(len(targets)), targets].mean()
    assert real > np.quantile(null.mean(axis=0), 0.95)


def test_matched_null_is_deranged_in_strata_and_reproducible(monkeypatch):
    monkeypatch.setitem(m.PARAMETERS, "null_draws", 39)
    _, _, targets, vectors, events = strong_fixture()
    null, donors, audit = m.matched_event_null(events, vectors)
    assert audit.matched_control_available.all()
    assert not (donors == np.arange(len(events))[:, None]).any()
    for draw in range(39):
        np.testing.assert_array_equal(np.sort(donors[:, draw]), np.arange(len(events)))
    assert vectors[np.arange(len(events)), targets].mean() > np.quantile(null.mean(axis=0), 0.95)
    np.testing.assert_array_equal(donors, m.matched_event_null(events, vectors)[1])


def test_small_strata_are_reported_not_pooled_into_other_epochs(monkeypatch):
    monkeypatch.setitem(m.PARAMETERS, "null_draws", 9)
    _, _, _, vectors, events = strong_fixture(8)
    events.loc[:2, "rest_epoch"] = 4
    null, donors, audit = m.matched_event_null(events, vectors)
    assert not audit.matched_control_available.iloc[:3].any()
    assert np.isnan(null[:3]).all() and (donors[:3] == -1).all()
    assert audit.matched_control_available.iloc[3:].all()
    assert (donors[3:] >= 3).all()


def test_frozen_map_or_selection_mutation_rejected(tmp_path):
    path = tmp_path / "selection.csv"
    path.write_text("frozen\n")
    manifest = {
        "parameters": copy.deepcopy(m.PARAMETERS),
        "run_parameters": copy.deepcopy(m.RUN_PARAMETERS),
        "code_commit": "x" * 40,
        "candidate_selection_uses_pfc_spikes": False,
        "maps": [],
        "frozen_files": {"selection.csv": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    m.validate_frozen(tmp_path, manifest, "x" * 40)
    with pytest.raises(ValueError, match="revision"):
        m.validate_frozen(tmp_path, manifest, "y" * 40)
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="hash"):
        m.validate_frozen(tmp_path, manifest, "x" * 40)


def test_coverage_completeness_checks_each_event_settings():
    rows = [
        {"event_id": e, "fraction": f, "repeat": r, "rule": rule}
        for e in ("a", "b")
        for f, r in [(1.0, -1)] + [(f, r) for f in (0.75, 0.5, 0.25) for r in range(20)]
        for rule in m.RULES
    ]
    frame = pd.DataFrame(rows)
    assert m.coverage_complete(frame, ["a", "b"])
    frame.loc[0, "repeat"] = 999
    assert not m.coverage_complete(frame, ["a", "b"])


def test_primary_summary_counts_events_not_thinning_repeats(monkeypatch):
    monkeypatch.setitem(m.PARAMETERS, "null_draws", 39)
    _, _, targets, vectors, events = strong_fixture(48)
    null, _, audit = m.matched_event_null(events, vectors)
    events["animal"] = "JS14"
    events["full_geometric_pass"] = True
    events["primary_lost_half"] = True
    events["matched_control_available"] = audit.matched_control_available
    events["pfc_reference_score"] = vectors[np.arange(48), targets]
    events["rest_key"] = "synthetic:2"
    result = m.group_summary(events, null, null)
    primary = result[result.group == "primary_lost_half"]
    assert (primary.n_events == 48).all()
    assert (primary.n_rest_epochs == 1).all()
    assert primary.above_control_p95.all()


def label_rows(events, full_pass=True):
    rows = []
    for event in events.itertuples():
        settings = [(1.0, -1)] + [(f, repeat) for f in (0.75, 0.5, 0.25) for repeat in range(m.PARAMETERS["repetitions"])]
        for fraction, repeat in settings:
            for rule in m.RULES:
                rows.append(
                    {
                        "event_id": event.event_id,
                        "animal": event.animal,
                        "fraction": fraction,
                        "repeat": repeat,
                        "rule": rule,
                        "geometric_pass": bool(full_pass and fraction == 1.0),
                    }
                )
    return pd.DataFrame(rows)


def test_full_report_separates_promising_pilot_from_paper_claim(tmp_path, monkeypatch):
    monkeypatch.setitem(m.PARAMETERS, "null_draws", 39)
    _, _, targets, vectors, events = strong_fixture(48)
    events["animal"] = ["JS14"] * 24 + ["ZT2"] * 24
    events["file"] = events.animal
    events["rest_epoch"] = np.tile(np.repeat([2, 4], 12), 2)
    events["pfc_reference_score"] = vectors[np.arange(48), targets]
    pair, _, _ = m.matched_event_null(events, vectors)
    labels = label_rows(events)
    assert m.summarize_scoring(events, labels, pair, pair, tmp_path)
    import json

    decision = json.loads((tmp_path / "decision.json").read_text())
    assert decision["decision"] == "promising_two_animal_pilot"
    assert not decision["paper_ready"]
    summary = pd.read_csv(tmp_path / "by_animal_content.csv")
    primary = summary[summary.group == "primary_lost_half"]
    assert (primary.n_events == 24).all()


def test_empty_primary_group_is_not_a_vacuous_pass(tmp_path, monkeypatch):
    monkeypatch.setitem(m.PARAMETERS, "null_draws", 9)
    _, _, targets, vectors, events = strong_fixture(48)
    events["animal"] = ["JS14"] * 24 + ["ZT2"] * 24
    events["file"] = events.animal
    events["rest_epoch"] = np.tile(np.repeat([2, 4], 12), 2)
    events["pfc_reference_score"] = vectors[np.arange(48), targets]
    pair, _, _ = m.matched_event_null(events, vectors)
    assert m.summarize_scoring(events, label_rows(events, full_pass=False), pair, pair, tmp_path)
    import json

    result = json.loads((tmp_path / "decision.json").read_text())
    assert result["decision"] == "underpowered_primary_group"
    assert not result["paper_ready"]
    gates = pd.read_csv(tmp_path / "gate_summary.csv").set_index("gate").passed
    assert not gates.primary_group_sufficient
    assert not gates.primary_both_controls_both_animals_positive
