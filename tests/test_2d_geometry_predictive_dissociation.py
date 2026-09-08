import numpy as np
import pandas as pd
import pytest

from scripts import audit_2d_geometry_predictive_dissociation as analysis


def test_overlapping_frames_retain_complete_exposure_only():
    counts = np.arange(18).reshape(9, 2)
    duration = np.r_[np.full(8, 0.005), 0.002]
    frames = analysis.frame_counts(counts, duration)
    np.testing.assert_array_equal(frames, np.stack([counts[i : i + 4].sum(axis=0) for i in range(5)]))
    np.testing.assert_array_equal(analysis.frame_counts(counts, np.full(9, 0.005)), np.stack([counts[i : i + 4].sum(axis=0) for i in range(6)]))
    assert analysis.frame_counts(counts[:3], np.full(3, 0.005)).shape == (0, 2)


@pytest.mark.parametrize("duration", [[0.005, 0], [0.005, 0.006], [0.004, 0.005], [0.005, np.nan]])
def test_bad_exposure_rejected(duration):
    with pytest.raises(ValueError):
        analysis.frame_counts(np.ones((2, 2)), np.asarray(duration))


def path(x):
    return np.column_stack([np.asarray(x, float), np.zeros(len(x))])


def test_continuous_static_and_jumpy_examples():
    counts = np.full((10, 2), 2)
    assert analysis.screen(path(np.arange(10) * 8), counts)["geometric_pass"]
    assert not analysis.screen(path(np.zeros(10)), counts)["geometric_pass"]
    assert not analysis.screen(path(np.arange(10) * 20), counts)["geometric_pass"]
    assert analysis.screen(path(np.arange(10) * 8), counts, minimum=11)["failure_reason"] == "insufficient_duration_or_support"


def test_strict_jump_inclusive_displacement_and_earliest_tie():
    counts = np.full((11, 2), 2)
    assert analysis.screen(path(np.arange(11) * 4), counts)["geometric_pass"]
    assert not analysis.screen(path(np.arange(11) * 3.999), counts)["geometric_pass"]
    positions = np.r_[np.zeros(10), 100 + np.arange(10) * 8]
    r = analysis.screen(path(positions), np.full((20, 2), 2))
    assert r["continuous_start"] == 0
    assert r["continuous_frames"] == 10
    assert not r["geometric_pass"]


def test_internal_unsupported_bins_do_not_bridge_filtered_runs():
    counts = np.full((12, 2), 2)
    counts[6] = 0
    positions = path(np.arange(12) * 8)
    assert analysis.screen(positions, counts)["geometric_pass"]
    result = analysis.screen(positions, counts, filtered=True)
    assert not result["geometric_pass"]
    assert result["continuous_frames"] == 6
    assert result["failure_reason"] == "jumps_or_unsupported_gaps"


def test_no_support_or_no_frames_fail():
    for n in (0, 20):
        result = analysis.screen(path(np.zeros(n)), np.zeros((n, 2), int))
        assert not result["geometric_pass"]
        assert result["continuous_frames"] == 0


def test_heldout_counts_and_rates_cannot_change_training_labels():
    centers = path(np.arange(10) * 8)
    rates = np.full((12, 10), 0.1)
    rates[:10] += np.eye(10) * 100
    counts = np.zeros((10, 12), int)
    counts[:, :10] = np.eye(10, dtype=int) * 6
    training = np.arange(10)
    original = analysis.classify_training(counts, rates, centers, training)
    assert original[0]["geometric_pass"]
    counts[:, 10:] = 10000
    rates[10:] = 10000
    assert analysis.classify_training(counts, rates, centers, training) == original
    assert not any(r["heldout_used_for_label"] for r in original)


def summary():
    rows = []
    for dataset, (events, sessions, animals) in analysis.EXPECTED.items():
        for stratum in ("all", "geometric_rejected"):
            for m in analysis.METRICS:
                rows.append(
                    {
                        "dataset": dataset,
                        "split": 0,
                        "criterion": "edge10",
                        "stratum": stratum,
                        "metric": m,
                        "events": events,
                        "contributing_sessions": sessions,
                        "contributing_animals": animals,
                        "positive_animals": animals,
                        "mean": 1.0,
                        "ci_low": 0.5,
                    }
                )
    return pd.DataFrame(rows)


def test_composition_failure_is_not_rescued_by_order():
    table = summary()
    table.loc[table.dataset.eq("tanni2022") & table.metric.eq("imm_minus_event_global"), "ci_low"] = -0.01
    result = analysis.decision(table).set_index("dataset")
    assert result.rejected_order_adjacency_supported.all()
    assert not result.loc["tanni2022", "rejected_beyond_other_event_composition_supported"]
    assert not result.biological_false_negative_rate_identified.any()
    assert not result.novel_mechanism_established.any()


@pytest.mark.parametrize("field,value", [("events", 0), ("contributing_animals", 0), ("positive_animals", 0), ("ci_low", -0.1)])
def test_empty_or_failed_rejected_group_cannot_pass(field, value):
    table = summary()
    table.loc[table.stratum.eq("geometric_rejected"), field] = value
    assert not analysis.decision(table).rejected_order_adjacency_supported.any()


def test_missing_primary_fails():
    with pytest.raises(ValueError):
        analysis.decision(summary().iloc[1:])


def test_strata_keep_empty_sessions_and_splits_do_not_mix(monkeypatch):
    monkeypatch.setattr(analysis, "EXPECTED", {d: (2, 2, 1) for d in analysis.EXPECTED})
    scores, labels = [], []
    for dataset in analysis.EXPECTED:
        for session in ("s1", "s2"):
            for split in range(5):
                key = {"dataset": dataset, "animal": "a", "session": session, "event_id": 1, "split": split}
                scores.append(key | {v: float(split + 1) for v in analysis.VALUES})
                for criterion in analysis.CRITERIA:
                    labels.append(key | {"criterion": criterion, "geometric_pass": session == "s1" and split != 1, "n_train_cells": 10, "heldout_used_for_label": False})
    result, sessions, _animals = analysis.summarize(pd.DataFrame(labels), pd.DataFrame(scores), draws=5)
    p = result[result.split.eq(0) & result.criterion.eq("edge10") & result.stratum.eq("geometric_rejected")]
    assert p.events.eq(1).all() and p["mean"].eq(1).all()
    q = result[result.split.eq(1) & result.stratum.eq("geometric_rejected")]
    assert q.events.eq(2).all() and q["mean"].eq(2).all() and q.ci_low.isna().all()
    empty = sessions[sessions.split.eq(0) & sessions.session.eq("s1") & sessions.stratum.eq("geometric_rejected")]
    assert empty.events.eq(0).all()
    assert empty[analysis.VALUES].isna().all().all()
