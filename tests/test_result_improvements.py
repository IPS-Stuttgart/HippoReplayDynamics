from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.result_improvements import (
    add_candidate_support_quality_columns,
    circular_shift_spikes_session,
    hierarchical_bootstrap_ci,
    paired_sign_flip_p_value,
    posterior_calibration_summary,
    shuffle_spike_times_session,
    shuffle_well_labels,
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


def test_candidate_support_quality_accepts_array_like_min_log_mass() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "state-space-imm",
                "evidence_support": "truncated_full_grid",
                "diagnostic_state_space_imm_min_candidate_log_mass": np.asarray([np.nan, -0.02]),
            }
        ]
    )

    labelled = add_candidate_support_quality_columns(rows)

    assert labelled.loc[0, "candidate_min_log_mass"] == -0.02
    assert labelled.loc[0, "candidate_support_quality"] == "conservative_warning"


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


def test_shuffle_spike_times_keeps_clusterless_mark_times_aligned() -> None:
    session = _marked_session()
    shuffled = shuffle_spike_times_session(session, random_seed=4)

    assert shuffled.spike_marks is not None
    np.testing.assert_allclose(shuffled.spike_marks.times, shuffled.spikes[:, 0])
    np.testing.assert_array_equal(shuffled.spike_marks.cell_ids, shuffled.spikes[:, 1].astype(int))
    np.testing.assert_allclose(shuffled.spike_marks.marks[:, 0], shuffled.spikes[:, 1])


def test_circular_shift_reorders_clusterless_mark_rows_with_spikes() -> None:
    session = _marked_session()
    shifted = circular_shift_spikes_session(session, shift_s=1.5)

    assert shifted.spike_marks is not None
    np.testing.assert_allclose(shifted.spike_marks.times, shifted.spikes[:, 0])
    np.testing.assert_array_equal(shifted.spike_marks.cell_ids, shifted.spikes[:, 1].astype(int))
    np.testing.assert_allclose(shifted.spike_marks.marks[:, 0], shifted.spikes[:, 1])


def test_shuffle_well_labels_preserves_id_coordinate_tuples() -> None:
    frame = pd.DataFrame(
        {
            "event": [0, 1, 2, 3],
            "true_well_id": ["A", "B", "C", None],
            "true_well_x": [1.0, 2.0, 3.0, np.nan],
            "true_well_y": [10.0, 20.0, 30.0, np.nan],
        }
    )
    allowed_tuples = {
        ("A", 1.0, 10.0),
        ("B", 2.0, 20.0),
        ("C", 3.0, 30.0),
    }

    shuffled = shuffle_well_labels(frame, random_seed=0)

    observed_tuples = {
        tuple(row)
        for row in shuffled.loc[
            shuffled["true_well_id"].notna(),
            ["true_well_id", "true_well_x", "true_well_y"],
        ].itertuples(index=False, name=None)
    }
    assert observed_tuples == allowed_tuples
    assert shuffled.loc[3, ["true_well_id", "true_well_x", "true_well_y"]].isna().all()
    assert shuffled["event"].tolist() == frame["event"].tolist()


def _marked_session() -> ReplaySession:
    spikes = np.array(
        [
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=float,
    )
    spike_marks = SpikeMarkData(
        times=spikes[:, 0].copy(),
        marks=np.array(
            [
                [10.0, 1.0],
                [20.0, 2.0],
                [30.0, 3.0],
                [40.0, 4.0],
            ],
            dtype=float,
        ),
        source_file="synthetic",
        source_variable="marks",
        feature_names=("cell_id_proxy", "feature"),
        cell_ids=spikes[:, 1].astype(int),
        group_ids=np.array([1, 2, 3, 4], dtype=int),
    )
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("."),
        position=np.empty((0, 3), dtype=float),
        spikes=spikes,
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=spike_marks,
    )
