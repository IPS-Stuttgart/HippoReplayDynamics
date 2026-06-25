from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.encoding import LogEmissionTensor
from scripts.clean_imm_time_order_shuffle_control import (
    FIRST_ORDER_IMM,
    FRAGMENTED,
    permute_emission_time_bins,
    read_precomputed_scores,
    select_event_groups,
    write_outputs,
)
from scripts.audit_imm_fragmented_hypotheses import DIFFUSION, MOMENTUM_EXACT, STATIONARY


def test_time_order_shuffle_decisions_detect_clean_imm_advantage(tmp_path: Path) -> None:
    scores = pd.DataFrame(
        [
            *_event_scores("Rat1/Open1", 0, "clean_imm", original_delta=30.0, shuffle_deltas=[1.0, 2.0, 3.0, 4.0]),
            *_event_scores("Rat1/Open1", 1, "clean_imm", original_delta=15.0, shuffle_deltas=[5.0, 6.0, 7.0, 8.0]),
            *_event_scores("Rat2/Open1", 2, "imm_fragmented_ambiguous", original_delta=2.0, shuffle_deltas=[2.0, 2.5, 3.0, 3.5]),
        ]
    )

    outputs = write_outputs(scores, tmp_path, expected_n_shuffles=4)
    decisions = outputs["clean_imm_time_order_shuffle_decisions.csv"]
    gates = outputs["clean_imm_time_order_shuffle_gate_summary.csv"].set_index("gate")
    by_group = outputs["clean_imm_time_order_shuffle_by_group.csv"].set_index("event_group")

    assert len(decisions) == 3
    assert bool(gates.loc["clean_imm_median_time_order_advantage_positive", "passed"])
    assert bool(gates.loc["clean_imm_majority_original_above_shuffle_median", "passed"])
    assert bool(gates.loc["clean_imm_at_least_some_original_above_shuffle_p95", "passed"])
    assert bool(gates.loc["ambiguous_controls_lower_advantage_than_clean_imm", "passed"])
    assert not bool(gates.loc["manifest_written", "passed"])
    assert not bool(gates.loc["technical_overall", "passed"])
    assert float(by_group.loc["clean_imm", "median_time_order_advantage"]) > 0.0
    assert (tmp_path / "clean_imm_time_order_original_vs_shuffled_scatter.png").is_file()
    assert (tmp_path / "clean_imm_time_order_advantage_by_group.png").is_file()


def test_missing_shuffled_scores_fail_nonvacuous_gates(tmp_path: Path) -> None:
    scores = pd.DataFrame(
        [
            *_event_scores("Rat1/Open1", 0, "clean_imm", original_delta=30.0, shuffle_deltas=[1.0]),
        ]
    )

    outputs = write_outputs(scores, tmp_path, expected_n_shuffles=2)
    gates = outputs["clean_imm_time_order_shuffle_gate_summary.csv"].set_index("gate")

    assert not bool(gates.loc["all_shuffle_scores_present", "passed"])
    assert not bool(gates.loc["n_shuffles_complete", "passed"])
    assert not bool(gates.loc["required_models_complete", "passed"])
    assert not bool(gates.loc["technical_overall", "passed"])


def test_empty_scores_do_not_vacuously_pass(tmp_path: Path) -> None:
    outputs = write_outputs(pd.DataFrame(), tmp_path, expected_n_shuffles=3)
    gates = outputs["clean_imm_time_order_shuffle_gate_summary.csv"].set_index("gate")

    assert not bool(gates.loc["selected_events_present", "passed"])
    assert not bool(gates.loc["required_models_complete", "passed"])
    assert not bool(gates.loc["technical_overall", "passed"])


def test_precomputed_score_reader_fills_optional_columns(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "event_group": "clean_imm",
                "score_kind": "original",
                "shuffle_index": -1,
                "model": FIRST_ORDER_IMM,
                "log_evidence": 10.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "event_group": "clean_imm",
                "score_kind": "original",
                "shuffle_index": -1,
                "model": FRAGMENTED,
                "log_evidence": 0.0,
            },
        ]
    ).to_csv(path, index=False)

    loaded = read_precomputed_scores(path)

    assert "rat" in loaded.columns
    assert loaded["rat"].iloc[0] == "Rat1"
    assert loaded["status"].eq("success").all()
    assert "n_active_units" in loaded.columns


def test_select_event_groups_balances_clean_ambiguous_and_momentum() -> None:
    evidence = pd.DataFrame(
        [
            *_exact_core("Rat1/Open1", 0, fragmented=20.0, first_order=50.0, momentum=30.0),
            *_exact_core("Rat1/Open1", 1, fragmented=20.0, first_order=24.0, momentum=18.0),
            *_exact_core("Rat2/Open1", 2, fragmented=20.0, first_order=0.0, momentum=60.0),
            *_exact_core("Rat2/Open1", 3, fragmented=20.0, first_order=60.0, momentum=30.0),
        ]
    )

    selected = select_event_groups(
        evidence,
        margin_threshold=5.5,
        max_clean_imm=1,
        max_ambiguous=1,
        max_momentum_like=1,
        seed=4,
    )

    assert set(selected["event_group"]) == {"clean_imm", "imm_fragmented_ambiguous", "momentum_like"}
    assert len(selected) == 3


def test_permute_emission_time_bins_reorders_observations_not_time_grid() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        spike_counts=np.array([[1, 0], [0, 2], [3, 0]]),
        times=np.array([0.5, 1.5, 2.5]),
        dt=1.0,
        cell_ids=np.array([10, 11]),
        n_spikes=6,
        bin_durations=np.ones(3),
        transition_durations=np.ones(2),
    )

    shuffled = permute_emission_time_bins(emissions, np.array([2, 0, 1]))

    np.testing.assert_array_equal(shuffled.log_likelihood, emissions.log_likelihood[[2, 0, 1]])
    np.testing.assert_array_equal(shuffled.spike_counts, emissions.spike_counts[[2, 0, 1]])
    np.testing.assert_array_equal(shuffled.times, emissions.times)
    np.testing.assert_array_equal(shuffled.bin_durations, emissions.bin_durations)
    np.testing.assert_array_equal(shuffled.transition_durations, emissions.transition_durations)
    assert shuffled.metadata["time_order_control"] == "whole_bin_shuffle"


def _event_scores(session: str, event_index: int, event_group: str, *, original_delta: float, shuffle_deltas: list[float]) -> list[dict[str, object]]:
    rows = [
        _score(session, event_index, event_group, "original", -1, FIRST_ORDER_IMM, original_delta),
        _score(session, event_index, event_group, "original", -1, FRAGMENTED, 0.0),
    ]
    for shuffle_index, delta in enumerate(shuffle_deltas):
        rows.append(_score(session, event_index, event_group, "shuffle", shuffle_index, FIRST_ORDER_IMM, delta))
        rows.append(_score(session, event_index, event_group, "shuffle", shuffle_index, FRAGMENTED, 0.0))
    return rows


def _score(
    session: str,
    event_index: int,
    event_group: str,
    score_kind: str,
    shuffle_index: int,
    model: str,
    log_evidence: float,
) -> dict[str, object]:
    return {
        "status": "success",
        "failure_reason": "",
        "session": session,
        "rat": session.split("/", 1)[0],
        "event_index": event_index,
        "event_group": event_group,
        "score_kind": score_kind,
        "shuffle_index": shuffle_index,
        "model": model,
        "log_evidence": log_evidence,
        "duration_ms": 30.0,
        "n_time": 10,
        "n_spikes": 8,
        "n_active_units": 4,
        "runtime_s": 0.01,
    }


def _exact_core(session: str, event_index: int, *, fragmented: float, first_order: float, momentum: float) -> list[dict[str, object]]:
    return [
        _evidence(session, event_index, STATIONARY, 0.0),
        _evidence(session, event_index, DIFFUSION, 10.0),
        _evidence(session, event_index, FRAGMENTED, fragmented),
        _evidence(session, event_index, FIRST_ORDER_IMM, first_order),
        _evidence(session, event_index, MOMENTUM_EXACT, momentum),
    ]


def _evidence(session: str, event_index: int, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "log_evidence": log_evidence,
        "evidence_comparable": True,
    }
