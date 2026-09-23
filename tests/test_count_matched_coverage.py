"""Exact-count intervention invariants and event-balanced aggregation."""

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.independent_rejected_forecast import full_counts
from hipporeplayimm.training_continuity import overlapping_counts
from scripts.audit_count_matched_coverage import DRAWS, conditional_path, contrasts, match_counts, summarize


def test_match_preserves_fine_overlap_and_coarse_counts():
    base = np.random.default_rng(17).poisson(2, (23, 8))
    targets = base[:, :4].sum(axis=1)
    x = match_counts(base, targets, np.random.default_rng(8))
    assert (x <= base).all()
    np.testing.assert_array_equal(x.sum(axis=1), targets)
    dt = np.r_[np.repeat(0.005, 22), 0.003]
    np.testing.assert_array_equal(overlapping_counts(x, dt).sum(axis=1), overlapping_counts(base[:, :4], dt).sum(axis=1))
    np.testing.assert_array_equal(full_counts(x, dt)[0].sum(axis=1), full_counts(base[:, :4], dt)[0].sum(axis=1))


def test_match_deterministic_and_zero_or_complete():
    x = np.array([[3, 4, 2], [0, 0, 0], [4, 5, 3]])
    n = np.array([9, 0, 5])
    a = match_counts(x, n, np.random.default_rng(4))
    np.testing.assert_array_equal(a, match_counts(x, n, np.random.default_rng(4)))
    np.testing.assert_array_equal(a[:2], x[:2])


@pytest.mark.parametrize("x,n", [([[1, -1]], [0]), ([[1.5, 2]], [1]), ([[1, 2]], [4]), ([[1, 2]], [0.5]), ([[1, 2]], [np.nan]), ([[1, 2]], [1, 2])])
def test_invalid_match(x, n):
    with pytest.raises(ValueError):
        match_counts(x, n, np.random.default_rng(1))


def test_hypergeometric_mean_is_uniform_over_spikes():
    x = np.tile([2, 6, 12], (10000, 1))
    got = match_counts(x, np.full(len(x), 5), np.random.default_rng(12))
    np.testing.assert_allclose(got.mean(axis=0), [0.5, 1.5, 3], atol=0.04)


def test_conditional_decoder_ignores_population_rate_scale():
    x = np.array([[1, 3], [4, 0], [0, 0]])
    rates = np.array([[1.0, 10.0, 2.0], [10.0, 1.0, 3.0]])
    expected = (x @ np.log(rates / rates.sum(axis=0))).argmax(axis=1)
    np.testing.assert_array_equal(conditional_path(x, rates), expected)
    np.testing.assert_array_equal(conditional_path(x, rates * [100, 0.01, 3]), expected)
    assert conditional_path(np.empty((0, 2), int), rates).size == 0


def fixture_rows():
    rows = []
    for event in range(2):
        for split in range(2):
            for arm, draws in [("full", [-1]), ("half_neurons", [-1]), ("matched_spikes", range(DRAWS))]:
                for d in draws:
                    rows.append(
                        {
                            "dataset": "test",
                            "animal": "a",
                            "session": "s",
                            "event_id": event,
                            "split": split,
                            "arm": arm,
                            "draw": d,
                            "n_heldout_target_spikes": 10,
                            "n_target_bins": 8,
                            "n_inference_spikes": 5,
                            "edge_supported_frames": 12,
                            "geometric_pass": arm != "half_neurons",
                            "n_active_inference_cells": 3,
                            "score_dynamic": -10 if arm == "half_neurons" else -5,
                            "score_matched_own": -12,
                        }
                    )
    return pd.DataFrame(rows)


def test_event_and_split_balancing_not_repeat_pseudoreplication():
    paired = contrasts(fixture_rows())
    events, sessions, animals, summary, _ = summarize(paired)
    assert len(events[events.group.eq("all")]) == 2
    assert len(sessions[sessions.group.eq("all")]) == 1
    assert len(animals[animals.group.eq("all")]) == 1
    row = summary[(summary.group == "all") & (summary.metric == "predictive_per_spike__matched_minus_half")].iloc[0]
    assert row["mean"] == 0.5
    assert row.animals == 1


@pytest.mark.parametrize("kind", ["missing", "duplicate", "targets", "counts"])
def test_incomplete_or_unmatched_rows_fail(kind):
    x = fixture_rows()
    if kind == "missing":
        x = x.iloc[1:]
    elif kind == "duplicate":
        x = pd.concat([x, x.iloc[:1]])
    elif kind == "targets":
        x.loc[2, "n_heldout_target_spikes"] = 11
    else:
        x.loc[2, "n_inference_spikes"] = 6
    with pytest.raises(ValueError):
        contrasts(x)


def test_zero_target_spikes_not_counted_as_prediction_evidence():
    x = fixture_rows()
    x.loc[x.event_id.eq(0), "n_heldout_target_spikes"] = 0
    paired = contrasts(x)
    assert paired.loc[paired.event_id.eq(0), "predictive_per_spike__matched_minus_half"].isna().all()
    assert paired.loc[paired.event_id.eq(0), "geometric_pass__matched_minus_half"].eq(1).all()
