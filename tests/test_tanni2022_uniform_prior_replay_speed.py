from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_tanni2022_uniform_prior_replay_speed import (
    benjamini_hochberg,
    finalize_replay_gate,
    independent_sequence_metrics,
    model_label_summary,
    permutation_bank,
    sequence_enrichment_summary,
    subset_segments,
)


def _posterior_for_path(path_bins: list[int], n_space: int = 6) -> np.ndarray:
    posterior = np.full((len(path_bins), n_space), 0.01, dtype=float)
    posterior[np.arange(len(path_bins)), path_bins] = 0.95
    return posterior / posterior.sum(axis=1, keepdims=True)


def test_linear_order_score_detects_monotonic_but_not_scrambled_path() -> None:
    centers = np.column_stack((np.arange(6, dtype=float) * 10.0, np.zeros(6)))
    bank = permutation_bank(6, 9_999, seed=4)
    ordered, _ = independent_sequence_metrics(_posterior_for_path([0, 1, 2, 3, 4, 5]), centers, bank)
    scrambled, _ = independent_sequence_metrics(_posterior_for_path([0, 4, 1, 5, 2, 3]), centers, bank)

    assert ordered["linear_order_score_abs_spearman"] == 1.0
    assert ordered["shuffle_empirical_p"] < 0.01
    assert scrambled["linear_order_score_abs_spearman"] < 0.5
    assert scrambled["shuffle_empirical_p"] > 0.1


def test_sequence_metric_does_not_require_constant_velocity() -> None:
    centers = np.column_stack((np.array([0.0, 1.0, 4.0, 10.0, 25.0, 50.0]), np.zeros(6)))
    metrics, _ = independent_sequence_metrics(
        _posterior_for_path([0, 1, 2, 3, 4, 5]),
        centers,
        permutation_bank(6, 9_999, seed=8),
    )

    assert metrics["linear_order_score_abs_spearman"] == 1.0
    assert metrics["posterior_path_efficiency"] > 0.99


def test_bh_adjustment_is_monotone_in_original_order() -> None:
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))

    np.testing.assert_allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_replay_gate_applies_extent_before_fdr_and_deduplicates_overlap() -> None:
    events = pd.DataFrame(
        {
            "animal": ["A", "A", "B"],
            "session": ["S", "S", "T"],
            "event_index": [1, 2, 3],
            "peak_ripple_z": [5.0, 10.0, 8.0],
            "n_spikes": [10, 10, 10],
            "n_active_cells": [5, 5, 5],
            "window_start_time_s": [1.0, 1.1, 2.0],
            "window_end_time_s": [1.2, 1.3, 2.2],
            "principal_axis_extent_cm": [50.0, 60.0, 10.0],
            "shuffle_empirical_p": [0.001, 0.002, 0.0001],
            "linear_order_score_abs_spearman": [0.9, 0.8, 1.0],
        }
    )

    gated = finalize_replay_gate(
        events,
        min_spatial_extent_cm=32.0,
        fdr_alpha=0.05,
        source_overlap_gap_s=0.0,
    )

    assert gated["uniform_prior_ordered_replay"].tolist() == [False, True, False]
    assert gated["uniform_prior_ordered_nominal_p05"].tolist() == [False, True, False]
    assert gated["uniform_prior_ordered_replay_deduplicated"].sum() == 1
    assert gated["uniform_prior_ordered_nominal_p05_deduplicated"].sum() == 1
    assert gated.loc[gated["uniform_prior_ordered_replay_deduplicated"], "event_index"].iloc[0] == 2


def test_sequence_enrichment_uses_source_representatives_only() -> None:
    events = pd.DataFrame(
        {
            "animal": ["A", "A", "B", "B"],
            "source_group_representative": [True, False, True, True],
            "spatially_extended": [True, True, True, False],
            "uniform_prior_ordered_nominal_p05": [True, True, False, True],
        }
    )

    summary = sequence_enrichment_summary(events).set_index("scope")

    assert summary.loc["all_animals", "tested_source_events"] == 2
    assert summary.loc["all_animals", "nominal_p05_events"] == 1


def test_segment_subset_and_model_join_use_only_independent_selection() -> None:
    events = pd.DataFrame(
        {
            "animal": ["A", "A"],
            "session": ["S", "S"],
            "event_index": [1, 2],
            "uniform_prior_ordered_replay_deduplicated": [True, False],
        }
    )
    segments = pd.DataFrame(
        {
            "animal": ["A", "A", "A"],
            "session": ["S", "S", "S"],
            "event_index": [1, 1, 2],
            "segment_index": [0, 1, 0],
        }
    )
    decisions = pd.DataFrame(
        {
            "animal": ["A"],
            "session": ["S"],
            "event_index": [1],
            "best_model": ["first-order-imm"],
            "ordered_trajectory_confident": [True],
            "imm_confident_over_fragmented": [False],
        }
    )

    selected_segments = subset_segments(segments, events, "uniform_prior_ordered_replay_deduplicated")
    events["uniform_prior_ordered_nominal_p05_deduplicated"] = [True, False]
    joined, summary = model_label_summary(events, decisions)

    assert selected_segments["event_index"].unique().tolist() == [1]
    assert len(joined) == 1
    assert summary.set_index("scope").loc["fdr_independent_replay", "model_scored_events"] == 1
    assert summary.set_index("scope").loc["nominal_p05_sensitivity", "best_first-order-imm"] == 1
