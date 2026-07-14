from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.report_pfeiffer_heldout_imm_fragmented import (
    FRAGMENTED_MODEL,
    IMM_MODEL,
    build_split_decisions,
    event_medians,
    gate_summary,
    grouped_event_summary,
    leave_one_rat_out,
    rat_cluster_bootstrap,
    run,
    summarize_event_medians,
)


def _scores(
    *,
    rats: int = 4,
    events_per_rat: int = 2,
    splits: int = 3,
    heldout_delta: float = 4.0,
    train_delta: float = 8.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rat_index in range(rats):
        rat = f"Rat{rat_index + 1}"
        session = f"{rat}/day1"
        for event_index in range(events_per_rat):
            for split_index in range(splits):
                fragmented_train = 10.0 + split_index
                fragmented_heldout = 2.0 + event_index
                for model, train, heldout in (
                    (FRAGMENTED_MODEL, fragmented_train, fragmented_heldout),
                    (
                        IMM_MODEL,
                        fragmented_train + train_delta,
                        fragmented_heldout + heldout_delta,
                    ),
                ):
                    rows.append(
                        {
                            "status": "success",
                            "session": session,
                            "rat": rat,
                            "event_index": event_index,
                            "cell_split_index": split_index,
                            "cell_split_seed": 100 + split_index,
                            "model": model,
                            "train_log_likelihood": train,
                            "joint_log_likelihood": train + heldout,
                            "heldout_log_likelihood": heldout,
                            "train_cell_count": 14,
                            "test_cell_count": 6,
                            "train_spikes": 30,
                            "test_spikes": 12,
                            "n_time": 25,
                            "test_cell_fraction": 0.3,
                            "heldout_predictive_method": "joint_log_evidence_minus_train_log_evidence",
                            "heldout_replay_spikes_used_for_latent_inference": False,
                            "train_diagnostic_mean_trajectory_posterior_entropy": 2.0,
                            "joint_diagnostic_mean_trajectory_posterior_entropy": 2.1,
                        }
                    )
    return pd.DataFrame(rows)


def test_split_decisions_are_true_conditional_and_train_selected() -> None:
    scores = _scores(rats=1, events_per_rat=1, splits=2)
    decisions = build_split_decisions(
        scores,
        frozen_clean_events={("Rat1/day1", 0)},
    )

    assert len(decisions) == 2
    assert decisions["pair_complete"].all()
    assert decisions["conditional_identity_verified"].all()
    assert decisions["explicit_no_heldout_latent_use"].all()
    assert decisions["train_defined_clean_imm"].all()
    assert decisions["frozen_clean_imm"].all()
    assert np.allclose(decisions["heldout_delta_imm_minus_fragmented"], 4.0)
    assert np.allclose(decisions["train_delta_imm_minus_fragmented"], 8.0)


def test_event_medians_prevent_split_or_event_weighting() -> None:
    scores = _scores(rats=1, events_per_rat=2, splits=3)
    decisions = build_split_decisions(scores)
    first = decisions["event_index"].eq(0)
    decisions.loc[first, "heldout_delta_imm_minus_fragmented"] = [10.0, 10.0, 1000.0]
    decisions.loc[~first, "heldout_delta_imm_minus_fragmented"] = -1.0

    events = event_medians(decisions)
    primary = events[events["scope"].eq("all_events")]
    assert len(primary) == 2
    assert sorted(primary["heldout_delta_event_median"].tolist()) == [-1.0, 10.0]
    summary = summarize_event_medians(events)
    observed = summary.loc[
        summary["scope"].eq("all_events"),
        "median_event_heldout_delta",
    ].iloc[0]
    assert observed == 4.5


def test_train_defined_scope_does_not_use_heldout_outcome() -> None:
    scores = _scores(rats=1, events_per_rat=2, splits=2)
    decisions = build_split_decisions(scores)
    event_zero = decisions["event_index"].eq(0)
    decisions.loc[event_zero, "train_defined_clean_imm"] = True
    decisions.loc[event_zero, "heldout_delta_imm_minus_fragmented"] = -20.0
    decisions.loc[~event_zero, "train_defined_clean_imm"] = False
    decisions.loc[~event_zero, "heldout_delta_imm_minus_fragmented"] = 20.0

    events = event_medians(decisions)
    selected = events[events["scope"].eq("train_defined_clean_imm")]
    assert selected["event_index"].tolist() == [0]
    assert selected["heldout_delta_event_median"].iloc[0] == -20.0


def test_duplicate_or_missing_pair_rows_are_incomplete() -> None:
    scores = _scores(rats=1, events_per_rat=1, splits=2)
    duplicate = scores.iloc[[0]].copy()
    missing_mask = (
        scores["cell_split_index"].eq(1)
        & scores["model"].eq(IMM_MODEL)
    )
    malformed = pd.concat([scores[~missing_mask], duplicate], ignore_index=True)
    decisions = build_split_decisions(malformed)

    assert not decisions["pair_complete"].any()


def test_gate_uses_event_medians_and_three_of_four_rats() -> None:
    scores = _scores(rats=4, events_per_rat=2, splits=3)
    decisions = build_split_decisions(scores)
    events = event_medians(decisions)
    by_rat = grouped_event_summary(events, "rat")
    loro = leave_one_rat_out(events)
    bootstrap = rat_cluster_bootstrap(events, replicates=100, seed=2)
    gates = gate_summary(
        scores,
        decisions,
        events,
        bootstrap,
        by_rat,
        loro,
        expected_events=8,
        expected_splits=3,
        expected_rats=4,
        min_positive_rats=3,
    )

    assert gates.loc[gates["gate"].eq("overall"), "passed"].iloc[0]


def test_gate_fails_when_latent_provenance_or_split_is_missing() -> None:
    scores = _scores(rats=4, events_per_rat=1, splits=3).drop(
        columns=["heldout_replay_spikes_used_for_latent_inference"]
    )
    scores = scores[
        ~(
            scores["session"].eq("Rat1/day1")
            & scores["event_index"].eq(0)
            & scores["cell_split_index"].eq(2)
        )
    ]
    decisions = build_split_decisions(scores)
    events = event_medians(decisions)
    by_rat = grouped_event_summary(events, "rat")
    loro = leave_one_rat_out(events)
    bootstrap = rat_cluster_bootstrap(events, replicates=50, seed=3)
    gates = gate_summary(
        scores,
        decisions,
        events,
        bootstrap,
        by_rat,
        loro,
        expected_events=4,
        expected_splits=3,
        expected_rats=4,
        min_positive_rats=3,
    )

    assert not gates.loc[
        gates["gate"].eq("repeated_splits_complete"),
        "passed",
    ].iloc[0]
    assert not gates.loc[
        gates["gate"].eq("heldout_spikes_excluded_from_latent_inference"),
        "passed",
    ].iloc[0]
    assert not gates.loc[gates["gate"].eq("overall"), "passed"].iloc[0]


def test_cli_run_writes_required_outputs(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.csv"
    frozen_path = tmp_path / "frozen.csv"
    output = tmp_path / "report"
    _scores(rats=4, events_per_rat=1, splits=2).to_csv(scores_path, index=False)
    pd.DataFrame(
        {
            "session": ["Rat1/day1"],
            "event_index": [0],
        }
    ).to_csv(frozen_path, index=False)
    args = Namespace(
        scores=str(scores_path),
        frozen_clean_selection=str(frozen_path),
        map_specificity_by_rat=None,
        output_dir=str(output),
        margin_threshold=5.5,
        expected_events=4,
        expected_splits=2,
        expected_rats=4,
        min_positive_rats=3,
        bootstrap_replicates=20,
        seed=1,
    )

    run(args)

    required = [
        "pfeiffer_heldout_imm_fragmented_split_decisions.csv",
        "pfeiffer_heldout_imm_fragmented_event_medians.csv",
        "pfeiffer_heldout_imm_fragmented_scope_summary.csv",
        "pfeiffer_heldout_imm_fragmented_gate_summary.csv",
        "pfeiffer_heldout_imm_fragmented_manifest.json",
        "pfeiffer_heldout_imm_fragmented_report.md",
    ]
    assert all((output / name).exists() for name in required)
