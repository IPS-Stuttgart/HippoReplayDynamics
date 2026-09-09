"""Nonleakage, geometric boundary and complete-pair regression tests."""

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.training_continuity import (
    CRITERIA,
    IDENTITY,
    classification_group,
    classify_training,
    geometry,
    grouped_event_values,
    nested_half,
    overlapping_counts,
    poisson_map,
    predictive_contrasts,
    validate_labels,
)


def test_overlapping_windows_and_partial_edge():
    counts = np.arange(14).reshape(7, 2)
    dt = np.r_[np.full(6, 0.005), 0.003]
    result = overlapping_counts(counts, dt)
    assert np.array_equal(result, np.stack([counts[i : i + 4].sum(axis=0) for i in range(3)]))
    assert overlapping_counts(counts[:2], dt[:2]).shape == (0, 2)
    with pytest.raises(ValueError):
        overlapping_counts(counts, np.r_[0.003, np.full(6, 0.005)])
    with pytest.raises(ValueError):
        overlapping_counts(counts + 0.2, dt)


def test_poisson_silence_and_flat_prior():
    rates = np.array([[1.0, 10.0], [1.0, 20.0]])
    counts = np.array([[0, 0], [2, 1], [0, 1]])
    expected = (counts @ np.log(rates) - 0.02 * rates.sum(axis=0)).argmax(axis=1)
    assert np.array_equal(poisson_map(counts, rates), expected)
    assert poisson_map(counts[:1], rates)[0] == 0


def test_geometric_boundaries_and_no_shortcut():
    grid = np.column_stack([np.arange(11) * (40 / 9), np.zeros(11)])
    counts = np.full((10, 2), 2)
    assert geometry(np.arange(10), grid, counts)["geometric_pass"]
    assert not geometry(np.arange(10), grid, counts, min_frames=11)["geometric_pass"]
    jump = np.column_stack([np.arange(10) * 20.0, np.zeros(10)])
    assert geometry(np.arange(10), jump, counts)["failure_reason"] == "jump_fragmentation"
    short = grid * 0.99
    assert geometry(np.arange(10), short, counts)["failure_reason"] == "insufficient_displacement"
    counts[4:6] = 0
    assert geometry(np.arange(10), grid, counts)["geometric_pass"]
    assert not geometry(np.arange(10), grid, counts, filtered=True)["geometric_pass"]


def test_earliest_longest_tie_and_zero_support():
    grid = np.column_stack([np.r_[np.arange(10) * 4.0, 100 + np.arange(10) * 5.0], np.zeros(20)])
    counts = np.full((20, 2), 2)
    result = geometry(np.arange(20), grid, counts)
    assert result["longest_run_frames"] == 10 and result["run_start_frame"] == 0
    assert not result["geometric_pass"]
    assert geometry(np.arange(20), grid, np.zeros_like(counts))["failure_reason"] == "no_supported_edges"
    assert geometry(np.empty(0, int), grid, np.empty((0, 2), int))["failure_reason"] == "no_complete_windows"


def test_heldout_spikes_and_maps_cannot_change_classification():
    rng = np.random.default_rng(4)
    counts = rng.poisson(0.5, (28, 8))
    rates = rng.uniform(0.1, 15, (8, 20))
    grid = np.column_stack([np.arange(20) * 5.0, np.zeros(20)])
    tr = np.array([0, 2, 3, 7])
    held = np.array([1, 4, 5, 6])
    first = classify_training(counts, np.full(28, 0.005), rates, grid, tr, np.ones(20, bool))
    counts[:, held] = 99999
    rates[held] = 1e8
    second = classify_training(counts, np.full(28, 0.005), rates, grid, tr, np.ones(20, bool))
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_nested_subsets_are_train_only_and_stable():
    tr = np.arange(0, 20, 2)
    a = nested_half(tr, ("dataset", "animal", "session"), 0)
    assert len(a) == 5 and set(a) <= set(tr)
    assert np.array_equal(a, nested_half(tr[::-1], ("dataset", "animal", "session"), 0))
    assert not np.array_equal(a, nested_half(tr, ("dataset", "animal", "session"), 1))
    assert classification_group(True, False) == "lost_with_thinning"


def fixture():
    rows, scores, shuffled = [], [], []
    identity = dict(zip(IDENTITY, ("dataset", "animal", "session"), strict=True))
    for event in (0, 1):
        for split in range(5):
            key = identity | {"event_id": event, "split": split}
            for support in ("parent", "arena_clipped"):
                for filtered, minimum in CRITERIA:
                    rows.append(
                        key
                        | {
                            "support": support,
                            "bin_filter": "bin_support" if filtered else "edge_only",
                            "min_frames": minimum,
                            "full_training_pass": False,
                            "nested_half_pass": False,
                            "full_valid_frames": 20,
                            "group": "rejected_both",
                            "heldout_used_for_classification": False,
                        }
                    )
            for map_name in ("real", "population_code_permuted"):
                scores.append(
                    key
                    | {
                        "map": map_name,
                        "n_heldout_spikes": 0 if event == 0 else 5,
                        "n_train_spikes": 20,
                        "duration_s": 0.1,
                        "score_first_order_imm": -10 + split,
                        "score_diffusion": -11 + split,
                        "score_iid_position": -15,
                        "score_static_location": -20,
                        "score_event_global": -22,
                    }
                )
            for model in ("first_order_imm", "diffusion"):
                for contrast in ("real_order_advantage", "order_map_interaction"):
                    shuffled.append(key | {"contrast": model + "__" + contrast, "delta": float(split + 1)})
    return pd.DataFrame(rows), pd.DataFrame(scores), pd.DataFrame(shuffled)


def test_event_medians_and_zero_spike_retention():
    labels, scores, shuffled = fixture()
    selected = labels[IDENTITY + ["event_id"]].drop_duplicates()
    validate_labels(labels, selected)
    predictions = predictive_contrasts(scores, shuffled)
    events = grouped_event_values(labels, predictions)
    row = events.query(
        "event_id == 0 and group == 'rejected_with_opportunity' and contrast == 'imm_minus_iid' and support == 'parent' and bin_filter == 'edge_only' and min_frames == 10"
    ).iloc[0]
    assert row.delta == 7 and row.qualifying_splits == 5
    assert np.isnan(row.delta_per_heldout_spike)
    assert len(events[events.event_id == 0]) == len(events[events.event_id == 1])


def test_missing_duplicate_leaking_schema_fail_nonvacuously():
    labels, scores, shuffled = fixture()
    selected = labels[IDENTITY + ["event_id"]].drop_duplicates()
    for bad in (labels.iloc[:-1], labels.iloc[0:0], pd.concat([labels, labels.iloc[:1]])):
        with pytest.raises(ValueError):
            validate_labels(bad, selected)
    labels.loc[0, "heldout_used_for_classification"] = True
    with pytest.raises(ValueError):
        validate_labels(labels, selected)
    for bad in (scores.iloc[:-1], pd.concat([scores, scores.iloc[:1]])):
        with pytest.raises(ValueError):
            predictive_contrasts(bad, shuffled)
    with pytest.raises(ValueError):
        predictive_contrasts(scores, shuffled.iloc[:-1])


def test_missing_prediction_join_does_not_drop_events():
    labels, scores, shuffled = fixture()
    predictions = predictive_contrasts(scores, shuffled)
    with pytest.raises(ValueError):
        grouped_event_values(labels, predictions.iloc[:-1])


def test_independent_geometry_reference_agrees():
    from scripts.verify_training_continuity_prediction import reference_metrics

    rng = np.random.default_rng(42)
    grid = rng.uniform(0, 50, (20, 2))
    for length in (0, 8, 20, 50):
        counts = rng.poisson(0.9, (length, 6))
        path = rng.integers(0, 20, length)
        for filtered, minimum in CRITERIA:
            assert geometry(path, grid, counts, filtered, minimum) == reference_metrics(path, grid, counts, filtered, minimum)


def test_report_decisions_fail_missing_groups_and_negative_axis():
    from hipporeplayimm.training_continuity import PRIMARY_CONTRASTS, PRIMARY_GROUPS
    from scripts.report_training_continuity_prediction import decision_table

    rows = []
    for dataset, animals in [("pfeiffer_foster", 4), ("tanni2022", 5)]:
        for group in PRIMARY_GROUPS:
            for contrast in PRIMARY_CONTRASTS:
                rows.append({"dataset": dataset, "group": group, "contrast": contrast, "animals": animals, "events": 20, "ci_low": 0.1, "positive_animals": animals})
    data = pd.DataFrame(rows)
    assert decision_table(data).joint_predictive_support.all()
    data.loc[0, "ci_low"] = -0.1
    assert not decision_table(data).joint_predictive_support.iloc[0]
    assert not decision_table(data.iloc[0:0]).joint_predictive_support.any()
    assert not decision_table(data.iloc[:-1]).joint_predictive_support.iloc[-1]


def test_hierarchical_interval_and_equal_event_weights():
    from scripts.report_training_continuity_prediction import aggregate_points, hierarchical_interval

    labels, scores, shuffled = fixture()
    events = grouped_event_values(labels, predictive_contrasts(scores, shuffled))
    group = events[
        (events.group == "all") & (events.contrast == "imm_minus_iid") & (events.support == "parent") & (events.bin_filter == "edge_only") & (events.min_frames == 10)
    ].copy()
    interval = hierarchical_interval(group, draws=30)
    assert np.allclose(interval[:, 0], 7)
    assert np.allclose(interval[:, 1], 1.4)
    group.loc[group.event_id == 0, "qualifying_splits"] = 1
    _, _, summary = aggregate_points(group)
    assert summary["mean"].iloc[0] == 7
