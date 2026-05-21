from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.result_improvements import (
    add_candidate_support_quality_columns,
    hierarchical_bootstrap_ci,
    paired_sign_flip_p_value,
    posterior_calibration_summary,
    stratified_cell_split,
)


def test_candidate_support_quality_labels_truncated_rows() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "state-space-imm",
                "evidence_support": "truncated_full_grid",
                "diagnostic_state_space_imm_min_candidate_log_mass": -0.005,
            },
            {
                "model": "state-space-imm",
                "evidence_support": "truncated_full_grid",
                "diagnostic_state_space_imm_min_candidate_log_mass": -1.0,
            },
        ]
    )
    labelled = add_candidate_support_quality_columns(rows)
    assert labelled.loc[0, "candidate_support_quality"] == "conservative_good"
    assert labelled.loc[1, "candidate_support_quality"] == "conservative_poor"


def test_hierarchical_bootstrap_ci_returns_interval() -> None:
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s2", "s2"],
            "model": ["imm", "imm", "imm", "imm"],
            "delta_vs_best_static": [1.0, 2.0, 3.0, 4.0],
        }
    )
    lo, hi = hierarchical_bootstrap_ci(
        rows,
        model="imm",
        n_bootstrap=100,
        random_seed=0,
    )
    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= hi


def test_paired_sign_flip_p_value_is_probability() -> None:
    rows = pd.DataFrame(
        {
            "model": ["imm"] * 5,
            "delta_vs_best_static": [1.0, 1.0, -0.5, 2.0, 0.5],
        }
    )
    p_value = paired_sign_flip_p_value(rows, model="imm", n_permutations=100, random_seed=0)
    assert 0.0 <= p_value <= 1.0


def test_stratified_cell_split_keeps_train_and_test_disjoint() -> None:
    cells = np.arange(12)
    scores = np.linspace(0.0, 1.0, cells.size)
    train, test = stratified_cell_split(cells, scores, 0.25, 1, n_strata=4)
    assert train.size + test.size == cells.size
    assert test.size > 0
    assert np.intersect1d(train, test).size == 0


def test_posterior_calibration_summary() -> None:
    samples = pd.DataFrame(
        {
            "session": ["s1", "s1"],
            "true_bin_probability": [0.5, 0.25],
            "true_bin_rank": [1, 2],
            "n_position_bins": [10, 10],
        }
    )
    summary = posterior_calibration_summary(samples)
    assert summary.loc[0, "rows"] == 2
    assert summary.loc[0, "mean_true_negative_log_probability"] > 0.0
